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

# Initialize the client with your API key
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Define the system prompt
SYSTEM_PROMPT = """You are the Imperial College London Chatbot, "Impy", designed to assist students and staff (the user) with various queries including but not limited to the Success Guide, timetables, and assignment due dates.

Your responses should be helpful, friendly, and tailored to the Imperial College community. You speak in British English, no z's!

Each message you receive will be structured in three parts:
1. Relevant information extracted from the database based on the user's question.
2. A summary of the conversation history (your working memory).
3. The user's new question or prompt.

Use the database information to provide accurate, up-to-date answers. The conversation history helps you maintain context and continuity. Always address the user's new question directly.

If the conversation history is empty, it's a new conversation. In this case, greet the user appropriately.

If relevant information is not provided in the database extract, don't mention this lack of information. Instead, provide advice based on your general knowledge of Imperial College, and suggest the user consult specific experts in their department (e.g., professors, GTAs, lab technicians) for more detailed information.

When possible, provide relevant URLs, especially if you're unsure or the information isn't available in the database extract. The main source of general information is https://www.imperial.ac.uk

If the user seems distressed, emphasize available help resources and offer empathetic support.

Remember, your goal is to provide helpful, accurate, and context-appropriate responses to assist the Imperial College community."""

# DATA ZONE #
class DataEntry:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata

class VectorDB:
    def __init__(self, name):
        self.name = name
        self.data = []
        self.vectorizer = TfidfVectorizer()
        self.index = None

    def add_entries(self, entries):
        self.data.extend(entries)
        self.build_index()
    
    def build_index(self):
        texts = [
            f"{entry.content} {' '.join(str(v) for v in entry.metadata.values())}"
            for entry in self.data
        ]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(tfidf_matrix.shape[1]))
        self.index.add_with_ids(tfidf_matrix.toarray().astype('float32'), np.array(range(len(texts))))
        
    def search(self, query, k=3):
        full_query = f"{query} {' '.join(str(v) for v in self.data[0].metadata.keys())}"
        query_vector = self.vectorizer.transform([full_query]).toarray().astype('float32')
        scores, indices = self.index.search(query_vector, k)
        results = [(self.data[i], score) for i, score in zip(indices[0], scores[0])]
        return results

class MultiDB:
    def __init__(self):
        self.databases = {}
        self.last_used_db = None
        self.db_configs = {
            'success_guide_ug': {
                'keywords': ['success', 'guide', 'advice', 'undergraduate'],
                'file_name': "success_guide_ug.json",
                'weight_multiplier': 1.5
            },
            'success_guide_pgt': {
                'keywords': ['success', 'guide', 'advice', 'taught postgraduate', 'masters', 'master'],
                'file_name': "success_guide_pgt.json",
                'weight_multiplier': 1.5
            },
            'success_guide_pgr': {
                'keywords': ['success', 'guide', 'advice', 'research', 'PhD', 'research'],
                'file_name': "success_guide_pgr.json",
                'weight_multiplier': 1.5
            },
            'union': {
                'keywords': ['club', 'clubs', 'society', 'societies', 'ECA', 'ECAs', 'extra-curricular', 'activities', 'activity'],
                'file_name': "imperial_union_data.json",
                'weight_multiplier': 1.5
            },
            'accommodation': {
                'keywords': ['accommodation', 'housing', 'house', 'houses', 'accomodation'],
                'file_name': "imperial_accommodation.json",
                'weight_multiplier': 150
            },
            'food': {
                'keywords': ['food', 'drink', 'eat', 'meal', 'coffee', 'tea'],
                'file_name': "imperial_food.json",
                'weight_multiplier': 50
            },
            # Add more database configurations here
        }

    def load_databases(self, base_path):
        base_path = os.path.join(os.path.dirname(__file__), base_path)
        
        for db_name, config in self.db_configs.items():
            file_path = os.path.join(base_path, config['file_name'])
            if os.path.exists(file_path):
                self.add_database(db_name, file_path)
            else:
                print(f"No {db_name} data found at {file_path}")

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
        prompt_words = set(prompt.lower().split())
        all_results = []
        
        for db_name, config in self.db_configs.items():
            if db_name not in self.databases:
                continue
            
            weight = config['weight_multiplier'] if any(keyword in prompt_words for keyword in config['keywords']) else 1.0
            
            results = self.databases[db_name].search(prompt, k=k)
            
            weighted_results = [(db_name, result, score * weight) for result, score in results]
            all_results.extend(weighted_results)
        
        all_results.sort(key=lambda x: x[2], reverse=True)
        
        top_results = all_results[:k]
        
        formatted_results = [
            (db_name, result, score) for db_name, result, score in top_results
        ]
        
        return "all", formatted_results

    def add_new_database(self, name, keywords, file_name, weight_multiplier=1.5):
        self.db_configs[name] = {
            'keywords': keywords,
            'file_name': file_name,
            'weight_multiplier': weight_multiplier
        }
        file_path = os.path.join(os.path.dirname(__file__), 'database', file_name)
        self.add_database(name, file_path)

    def reset_context(self):
        self.last_used_db = None

# CONVERSATION MANAGER #

class ConversationManager:
    def __init__(self, max_memory_length=1500):
        self.memory = ""
        self.current_exchange = {"user": "", "assistant": ""}
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.max_memory_length = max_memory_length

    def update(self, user_input: str, bot_response: str):
        self.current_exchange["user"] = user_input
        self.current_exchange["assistant"] = bot_response
        summary = self.generate_summary()
        self.memory = self.add_to_memory(summary)

    def generate_summary(self):
        prompt = f"""Summarize the following exchange in one concise sentence, as if you're explaining to Claude (the AI) what was discussed. Use "you" to refer to Claude/Impy, and "the user" for the human. Focus on the key points and any changes in topic:

User: {self.current_exchange['user']}
Assistant: {self.current_exchange['assistant']}

Summary:"""

        try:
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                system="You are a summarization assistant. In one sentence, provide a concise summary of the given exchange, written as if explaining to an AI what was just discussed.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return ""

    def add_to_memory(self, new_summary):
        if len(self.memory) + len(new_summary) <= self.max_memory_length:
            return f"{self.memory} {new_summary}".strip()
        else:
            # Split the memory into sentences
            sentences = re.split(r'(?<=[.!?])\s+', self.memory)
            
            # Remove sentences from the beginning until there's room for the new summary
            while len(' '.join(sentences)) + len(new_summary) > self.max_memory_length:
                sentences.pop(0)
            
            # Join the remaining sentences and add the new summary
            return f"{' '.join(sentences)} {new_summary}".strip()

    def get_context(self):
        return self.memory

# Claude time #

def get_claude_response(prompt, multi_db, conversation_manager, is_regenerate=False):
    # Helps find the relevant stuff based on the prompt
    result = multi_db.analyze_and_search(prompt)
    db_name, relevant_info = result[0], result[1]
    
    # Adds all the relevant info into a empty string called formatted_info
    formatted_info = ""
    if db_name and relevant_info:
        formatted_info = f"Relevant information from {db_name.upper()}:\n"
        for entry in relevant_info:
            formatted_info += f"- Content: {entry.content}\n"
            formatted_info += f"  Metadata: {json.dumps(entry.metadata, indent=2)}\n"
            for key, value in entry.metadata.items():
                if str(value).lower() in prompt.lower():
                    formatted_info += f"  Matched metadata: {key}: {value}\n"

    # Calls baby claude to give it its memory
    context = conversation_manager.get_context()

    try:
        regenerate_instruction = """
        The user has requested a different answer to their previous question. 
        They were not satisfied with the initial response. 
        Please generate a new, alternative answer to the same question. 
        Provide different information, a new perspective, or elaborate on aspects 
        not covered in the initial response. Ensure this new answer is distinct 
        from the previous one while still accurately addressing the user's question.
        """ if is_regenerate else ""
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"{formatted_info}\n\nConversation context:\n{context}\n\nUser's question: {prompt}\n\n{regenerate_instruction}"}
            ]
        )
        response = message.content[0].text
        response = format_hyperlinks(response)

        # Always update the conversation manager
        conversation_manager.update(prompt, response)
        
        return response
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
    url_pattern = re.compile(r'(https?://[^\s]+)')
    formatted_text = url_pattern.sub(r'<a href="\1" target="_blank">\1</a>', text)
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
            user_message = data.get('message', '')
            is_regenerate = data.get('is_regenerate', False)
            print(f"DEBUG: Received message: {user_message}, Regenerate: {is_regenerate}")
            
            # Initialize the database and conversation manager
            multi_db = MultiDB()
            multi_db.load_databases('database')

            conversation_manager = ConversationManager()
            if 'conversation_manager' in request.session:
                conversation_manager.memory = request.session['conversation_manager']
            
            # Get Claude response
            response = get_claude_response(user_message, multi_db, conversation_manager, is_regenerate=is_regenerate)

            # Always update the conversation manager
            request.session['conversation_manager'] = conversation_manager.get_context()
            request.session.modified = True

            # Stream the response
            return StreamingHttpResponse(generate_streamed_response(response), content_type='text/plain')

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
