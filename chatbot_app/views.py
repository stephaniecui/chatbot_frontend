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

# USER PROFILING #
# This function is to ask the user for initial information on year and course.
# If we have access to the year and course automatically via login this is where we would change the variable.
def initialize_user_profile():
    level = input("What is your level of study? (ug for undergraduate, pgt for masters, pgr for PhD): ")
    return {"level": level}

# DATA ZONE #
class DataEntry:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata

class VectorDB:
    def __init__(self, name: str):
        self.name = name
        self.data: List[DataEntry] = []
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
        
    def search(self, query, k=5):
        full_query = f"{query} {' '.join(str(v) for v in self.data[0].metadata.keys())}"
        query_vector = self.vectorizer.transform([full_query]).toarray().astype('float32')
        _, indices = self.index.search(query_vector, k)
        return [self.data[i] for i in indices[0]]

class MultiDB:
    def __init__(self, user_profile):
        self.databases = {}
        self.user_profile = user_profile
        self.last_used_db = None

    def load_databases(self, base_path):
        # Ensure base_path is correct
        base_path = os.path.join(settings.BASE_DIR, base_path)
        
        # Load success guide database based on user's level of study
        level = self.user_profile['level']
        success_guide_file = f"success_guide_{level}.json"
        success_guide_path = os.path.join(base_path, success_guide_file)
        
        if os.path.exists(success_guide_path):
            self.add_database('success_guide', success_guide_path)
        else:   
            print(f"No success guide found for {level} students")

       # DEBUG: Print loaded databases for debugging
        print(f"Loaded databases: {success_guide_file}")
    
    def add_database(self, name, file_path):
        db = VectorDB(name)
        with open(file_path, 'r') as f:
            data = json.load(f)
        entries = [DataEntry(item['content'], item.get('metadata', {})) for item in data]
        db.add_entries(entries)
        self.databases[name] = db
       
    def analyze_and_search(self, prompt, k=5):
        # Always use the success guide database
        db_name = 'success_guide'
        
        if db_name in self.databases:
            self.last_used_db = db_name
            return db_name, self.databases[db_name].search(prompt, k)
        else:
            return None, []

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

def get_claude_response(prompt, multi_db, conversation_manager, is_new_conversation=False):
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
        message = client.messages.create(
            # Currently using Claude's best model, efficient and powerful
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"{formatted_info}\n\nConversation context:\n{context}\n\nUser's new question: {prompt}"}
            ]
        )
        response = message.content[0].text
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
            time.sleep(0.05)  # Adjust the delay as needed
        yield '\n\n'  # Add a new paragraph

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
            print(f"DEBUG: Received message: {user_message}")

            if 'user_profile' not in request.session:
                # First interaction: prompt for level of study
                if not user_message:
                    return StreamingHttpResponse(generate_streamed_response("What is your level of study? (ug for undergraduate, pgt for masters, pgr for PhD): "), content_type='text/plain')
                else:
                    # Save user profile in session
                    request.session['user_profile'] = {"level": user_message}
                    request.session.modified = True
                    return StreamingHttpResponse(generate_streamed_response(f"Thank you. How may Impy help you today?"), content_type='text/plain')
            
            user_profile = request.session['user_profile']
            
            # Initialize the database and conversation manager
            multi_db = MultiDB(user_profile)
            multi_db.load_databases('database')

            conversation_manager = ConversationManager()
            if 'conversation_manager' in request.session:
                conversation_manager.memory = request.session['conversation_manager']
            
            # Get Claude response
            response = get_claude_response(user_message, multi_db, conversation_manager)

            request.session['conversation_manager'] = conversation_manager.get_context()
            request.session.modified = True

            # Stream the response
            return StreamingHttpResponse(generate_streamed_response(response), content_type='text/plain')

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
