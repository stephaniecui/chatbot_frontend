import os
import json
import time
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .models import ChatMessage
import re
import anthropic
from openai import OpenAI
import nltk  
from nltk.corpus import stopwords

# API clients
claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """Hello there! I'm Impy, the friendly Imperial College London Success Guide chatbot. My purpose is to be a helpful and approachable guide for students on the range of topics covered in the Imperial College London Success Guide. I speak in British English, no z's!

When you send me a message, I'll provide a response structured to directly address your specific question or query. I'll format my response using clear, easy-to-scan formatting such as:
* Bullet points
* Numbered lists
* Section headings
* Concise paragraphs 
I will avoid using the markdown-style headings and inline links.

First, I'll draw on relevant information from the database to directly address your specific question or query. Then, if a brief summary of our conversation so far would provide helpful context, I'll include that. But if the flow of the discussion is clear, I won't interrupt it with an unnecessary recap. Finally, I'll conclude by prompting you for your new question. This three-part structure will ensure my responses are tailored, informative and coherent. I won't introduce unrelated examples or tangents, as that could be confusing. Instead, I'll keep my focus squarely on being directly responsive to what you've asked.

If this is a brand new conversation, I'll greet you warmly and prompt you for your question or request. I won't make assumptions about your level of study. When the database doesn't have all the details, I'll do my best to provide helpful advice based on my knowledge of Imperial College London. I may suggest reaching out to specific experts (e.g., professors, GTAs, lab technicians) in your department for more specialised information.

Whenever possible, I'll include relevant website links, especially if I'm unsure about something or the info isn't in the database. The main Imperial College site at https://www.imperial.ac.uk is a great general resource.

Now, if you seem a bit stressed or worried about something, I'll be sure to emphasise the available support resources and offer a kind, empathetic ear. My goal is to be a friendly, knowledgeable, and reassuring presence for the Imperial community.

So, how can I assist you today? I'm excited to put my capabilities to work and help make your Imperial experience as smooth and successful as possible. Fire away with your question or request, and let's get started!"""

### DATA ZONE ###

# This is the knowledge base of Impy. If you see the relevant files that are next to Impy3.py in the directory
# you will find multiple .JSON files. They're structured like a dictionary with multiple entries. This part of
# the script helps find the most relevant data entries regarding the user's prompts to eventually feed the 
# Claude API. 

# The first class, defining a single data entry.
class DataEntry:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata

#
# Here's an example of a data entry in the JSON file:
#
# {
#     "content": "Computing in-Class test (MATE50001 Mathematics and Computing 2)",
#     "metadata": {
#       "course": "MATE50001 Mathematics and Computing 2",
#       "due_date": "18/11/23",
#       "submission_method": "In-person",
#       "percentage": "15",
#       "assignment_type": "Test",
#       "Year": "2"
#     }
#   }
#
# The benefit of this data structure is that metadata is not strictly defined in these categories.
# The success guide entries for example has a URL section, category for student status, etc.
# Claude is able to contextualise these categories to give the most appropriate response.
#

# The second class takes all those data entries (the combined content and metadata) and vectorises them with 
# the TfidfVectorizer function. Once that is done, we use FAISS to index everything for extremely quick searching.
# 
class VectorDB:
    def __init__(self, name):
        self.name = name
        self.data = []
        nltk.download('stopwords', quiet=True)
        self.stop_words = list(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        self.index = None

    def add_entries(self, entries):
        self.data.extend(entries)
        self.build_index()
    
    def preprocess_text(self, text):
        # Convert to lowercase and split into words
        words = text.lower().split()
        # Remove stop words and keep only words longer than 2 characters
        return ' '.join([word for word in words if word not in self.stop_words and len(word) > 2])

    def build_index(self):
        texts = [
            self.preprocess_text(f"{entry.content} {' '.join(str(v) for v in entry.metadata.values())}")
            for entry in self.data
        ]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(tfidf_matrix.shape[1]))
        self.index.add_with_ids(tfidf_matrix.toarray().astype('float32'), np.array(range(len(texts))))

    def search(self, query, k=3):
        preprocessed_query = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([preprocessed_query]).toarray().astype('float32')
        scores, indices = self.index.search(query_vector, k)
        results = [(self.data[i], score) for i, score in zip(indices[0], scores[0])]
        return results

# The third class helps load the relevant databases based on the course and year into a contextual database
# for the user's instance.

class MultiDB:
    def __init__(self):
        self.databases = {}
        self.last_used_db = None
        self.db_configs = {
            'success_guide_ug': {
                'keywords': ['success', 'guide', 'advice', 'undergraduate', 'undergrad'],
                'file_name': "success_guide_ug.json",
                'weight_addition': 1.5
            },
            'success_guide_pgt': {
                'keywords': ['success', 'guide', 'advice', 'taught postgraduate', 'masters', 'master', 'postgrad'],
                'file_name': "success_guide_pgt.json",
                'weight_addition': 1.5
            },
            'success_guide_pgr': {
                'keywords': ['success', 'guide', 'advice', 'research', 'PhD', 'research', 'postgrad'],
                'file_name': "success_guide_pgr.json",
                'weight_addition': 1.5
            },
            'union': {
                'keywords': ['club', 'clubs', 'society', 'societies', 'ECA', 'ECAs', 'extra-curricular', 'activities', 'activity'],
                'file_name': "imperial_union_data.json",
                'weight_addition': 1.5
            },
            'accommodation': {
                'keywords': ['accommodation', 'housing', 'house', 'houses'],
                'file_name': "imperial_accommodation.json",
                'weight_addition': 1.5
            },
            'food': {
                'keywords': ['food', 'drink', 'eat', 'meal', 'coffee', 'tea', 'eats','food', 'vegan', 'halal'],
                'file_name': "imperial_food.json",
                'weight_addition': 1.5
            },
            'funding': {
                'keywords': ['fees', 'fee', 'pay', 'price', 'funding', 'funds', 'fund'],
                'file_name': "imperial_funding.json",
                'weight_addition': 1.5
            },
        }

    def load_databases(self, base_path):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'database'))
        
        for db_name, config in self.db_configs.items():
            file_path = os.path.join(base_path, config['file_name'])
            if os.path.exists(file_path):
                self.add_database(db_name, file_path)
            else:
                print(f"No {db_name} data found at {file_path}")

        print(f"Loaded databases: {', '.join(self.databases.keys())}")

    def add_database(self, name, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            db = VectorDB(name)
            entries = [DataEntry(item['content'], item.get('metadata', {})) for item in data]
            db.add_entries(entries)
            self.databases[name] = db
        except FileNotFoundError:
            print(f"Database file not found: {file_path}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path}")

    def analyze_and_search(self, prompt, k=3):
        if not self.databases:
            return []
        
        original_prompt_words = set(prompt.lower().split())
        preprocessed_prompt = self.databases[next(iter(self.databases))].preprocess_text(prompt)
        all_results = []
                
        for db_name, config in self.db_configs.items():
            if db_name not in self.databases:
                continue
            
            matched_keywords = [keyword for keyword in config['keywords'] if keyword in original_prompt_words]
            weight = config['weight_addition'] if matched_keywords else 1.0
                
            results = self.databases[db_name].search(preprocessed_prompt, k=k)
            
            weighted_results = [(db_name, result, score + weight) for result, score in results]
            all_results.extend(weighted_results)
            
        # Sort results and remove duplicates based on content
        all_results.sort(key=lambda x: x[2], reverse=True)
        unique_results = []
        seen_content = set()
        for db_name, result, score in all_results:
            if result.content not in seen_content:
                unique_results.append((db_name, result, score))
                seen_content.add(result.content)
            if len(unique_results) == k:
                break
                
        return unique_results

    def add_new_database(self, name, keywords, file_name, weight_addition=1.5):
        self.db_configs[name] = {
            'keywords': keywords,
            'file_name': file_name,
            'weight_addition': weight_addition
        }
        file_path = os.path.join(os.path.dirname(__file__), 'database', file_name)
        self.add_database(name, file_path)

    def reset_context(self):
        self.last_used_db = None

# CONVERSATION MANAGER #

# This is Impy's memory. Without it, each new prompt would be like the first interaction.
# In the first iteration I just gave Impy the previous 3 exchanges, but this proved to be a lot of text.
# My workaround was to use a baby version of Claude (Haiku 3) to analyse the conversation and summarise the conversation.
# This gets added to working memory that contains the summary of the ongoing conversation.
# This memory is then inserted to the final message we give to the main Claude API.

class ConversationManager:
    def __init__(self, api_choice, max_memory_length=1000):
        self.memory = ""
        self.current_exchange = {"user": "", "assistant": ""}
        self.api_choice = api_choice
        self.max_memory_length = max_memory_length

    def update(self, user_input: str, bot_response: str):
        self.current_exchange["user"] = user_input
        self.current_exchange["assistant"] = bot_response
        summary = self.generate_summary()
        self.memory = self.add_to_memory(summary)

    def generate_summary(self):
        prompt = f"""Summarize the following exchange in one concise sentence, as if you're explaining to an AI what was discussed. Use "you" to refer to Impy, and "the user" for the human. Focus on the key points and any changes in topic:

User: {self.current_exchange['user']}
Assistant: {self.current_exchange['assistant']}

Summary:"""

        try:
            if self.api_choice == 'claude':
                message = claude_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=500,
                    system="You are a summarization assistant. In one sentence, provide a concise summary of the given exchange, written as if explaining to an AI what was just discussed.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return message.content[0].text.strip()
            else:  # OpenAI
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                    {"role": "system", "content": "You are a summarization assistant. In one sentence, provide a concise summary of the given exchange, written as if explaining to an AI what was just discussed."},
                    {"role": "user", "content": prompt}
                    ],
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return ""

    def add_to_memory(self, new_summary):
        if len(self.memory) + len(new_summary) <= self.max_memory_length:
            return f"{self.memory} {new_summary}".strip()
        else:
            sentences = re.split(r'(?<=[.!?])\s+', self.memory)
            while len(' '.join(sentences)) + len(new_summary) > self.max_memory_length:
                sentences.pop(0)
            return f"{' '.join(sentences)} {new_summary}".strip()

    def get_context(self):
        return self.memory

def get_api_response(prompt, multi_db, conversation_manager, is_regenerate=False):
    result = multi_db.analyze_and_search(prompt)
    
    formatted_info = "Relevant information from multiple databases:\n"
    for db_name, entry, score in result:
        formatted_info += f"- From {db_name.upper()} (score: {score:.4f}):\n"
        formatted_info += f"  Content: {entry.content}\n"
        if entry.metadata:
            formatted_info += f"  Metadata: {json.dumps(entry.metadata, indent=2)}\n"

    context = conversation_manager.get_context()
    
    full_prompt = f"{formatted_info}\n\nConversation context:\n{context}\n\nUser's new question: {prompt}"

    try:
        regenerate_instruction = """
        The user has requested a different answer to their previous question. 
        They were not satisfied with the initial response. 
        Please generate a new, alternative answer to the same question. 
        Provide different information, a new perspective, or elaborate on aspects 
        not covered in the initial response. Ensure this new answer is distinct 
        from the previous one while still accurately addressing the user's question.
        """ if is_regenerate else ""

        if conversation_manager.api_choice == 'claude':
            message = claude_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"{full_prompt}{regenerate_instruction}"}
                ]
            )
            response_text = message.content[0].text
        else:  # OpenAI
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{full_prompt}{regenerate_instruction}"}
                ],
                max_tokens=1000
            )
            response_text = response.choices[0].message.content

        if callable(format_hyperlinks):
            response_text = format_hyperlinks(response_text)
        
        conversation_manager.update(prompt, response_text)
        return response_text
    except Exception as e:
        return f"An error occurred: {str(e)}"

def generate_streamed_response(response):
    paragraphs = response.split('\n')
    for paragraph in paragraphs:
        words = paragraph.split()
        for word in words:
            yield word + ' '
            time.sleep(0.025)  # Adjust the delay as needed
        yield '\n\n'  # Add a new paragraph

def format_hyperlinks(text):
    url_pattern = re.compile(r'(https?://[^\s)]+)')
    formatted_text = url_pattern.sub(r'<a href="\g<0>" target="_blank">\g<0></a>', text)
    return formatted_text

def index(request):
    # Clear the session data each time the page is loaded
    request.session.flush()
    return render(request, 'chatbot_app/index.html')

@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            user_input = data.get('message', '').strip()
            print(f"DEBUG: Received message: {user_input}")

            # Initialize session if it's the first request
            if 'initialized' not in request.session:
                request.session['initialized'] = True
                request.session['is_new_conversation'] = True
                return StreamingHttpResponse(generate_streamed_response("Welcome to Impy, your Imperial College London assistant! Which API would you like to use? (A or B): "), content_type='text/plain')

            # Handle API choice if not set
            if 'api_choice' not in request.session:
                if user_input.upper() == 'A':
                    request.session['api_choice'] = 'claude'
                elif user_input.upper() == 'B':
                    request.session['api_choice'] = 'openai'
                else:
                    return StreamingHttpResponse(generate_streamed_response("Invalid choice. Please enter 'A' for Claude or 'B' for OpenAI: "), content_type='text/plain')
                
                request.session.modified = True
                return StreamingHttpResponse(generate_streamed_response(f"Type 'exit' to end the conversation. Hi, what would you like help on today?"), content_type='text/plain')

            # Handle 'exit' command
            if user_input.lower() == 'exit':
                request.session.flush()
                return StreamingHttpResponse(generate_streamed_response("Thank you for chatting with Impy. Goodbye!"), content_type='text/plain')

            # Process the user input
            api_choice = request.session['api_choice']
            is_new_conversation = request.session.get('is_new_conversation', True)

            multi_db = MultiDB()
            multi_db.load_databases('database')

            conversation_manager = ConversationManager(api_choice)
            if 'conversation_manager' in request.session:
                conversation_manager.memory = request.session['conversation_manager']

            response = get_api_response(user_input, multi_db, conversation_manager, is_new_conversation)

            # Update session
            request.session['conversation_manager'] = conversation_manager.get_context()
            request.session['is_new_conversation'] = False
            request.session.modified = True

            # Stream the response
            return StreamingHttpResponse(generate_streamed_response(response), content_type='text/plain')

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
