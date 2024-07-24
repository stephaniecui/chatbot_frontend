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
Your responses should be helpful, friendly, and tailored to the Imperial College community.
You will be given content extracted from a database based on the user's questions.
You speak in British English, no z's! 
You will be given a context window that summarises the conversation you've had so far. If it's empty, that means it's a new conversation.
This means if conversation context is not empty, you don't need to greet them again.
If the user is feeling distressed, emphasise on giving help resources.
If relevant information is not provided, you don't need to say you don't have specific information from the database, just advise them reasonably within context.
When this happens, also make sure to ask for more specifics within the experts in their department, like professors, GTAs, lab technicians, etc.
Give the relevant URLs when you can, especially if you're unsure or the information isn't available.
The main source of general information in the database is https://www.imperial.ac.uk"""

# USER PROFILING #
# This function is to ask the user for initial information on year and course.
# If we have access to the year and course automatically via login this is where we would change the variable.
def initialize_user_profile():
    year = input("What year are you in? (1/2/3/4): ")
    course = input("What is your course? (e.g., Materials): ")
    return {"year": year, "course": course}

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

    def add_entries(self, entries: List[DataEntry]):
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
        
        # Loads universal databases, information relevant to all Imperial student/staff.
        for file_name in os.listdir(base_path):
            if file_name.endswith('.json'):
                db_name = file_name[:-5]  # Remove .json extension
                file_path = os.path.join(base_path, file_name)
                self.add_database(db_name, file_path)

        # Load course-specific databases, based on the first asking of the user of course and year.
        course = self.user_profile['course']
        year = self.user_profile['year']
        course_path = os.path.join(base_path, course, f"Year {year}")
        
        if os.path.exists(course_path):
            for file_name in os.listdir(course_path):
                if file_name.endswith('.json'):
                    db_name = f"{course}_year{year}_{file_name[:-5]}"
                    file_path = os.path.join(course_path, file_name)
                    self.add_database(db_name, file_path)
        else:
            print(f"No specific data found for {course} Year {year}")

        # DEBUG: Print loaded databases for debugging
        print("Loaded databases:", list(self.databases.keys()))
    
    def add_database(self, name, file_path):
        db = VectorDB(name)
        with open(file_path, 'r') as f:
            data = json.load(f)
        entries = [DataEntry(item['content'], item.get('metadata', {})) for item in data]
        db.add_entries(entries)
        self.databases[name] = db
       
    def analyze_and_search(self, prompt, k=5):
        assignment_keywords = ['test', 'tests', 'coursework', 'exam', 'exams', 'assignment', 'assignments', 'deadline', 'due date']
        timetable_keywords = ['class', 'classes', 'lecture', 'lectures', 'seminar', 'lab', 'schedule', 'timetable']
        success_guide_keywords = ['success', 'tip', 'tips', 'guide', 'advice', 'help', 'study']

        prompt_lower = prompt.lower()
        
        if any(keyword in prompt_lower for keyword in assignment_keywords):
            db_name = f"{self.user_profile['course']}_year{self.user_profile['year']}_assignments"
        elif any(keyword in prompt_lower for keyword in timetable_keywords):
            db_name = f"{self.user_profile['course']}_year{self.user_profile['year']}_timetables"
        elif any(keyword in prompt_lower for keyword in success_guide_keywords):
            db_name = 'success_guide'
        else:
            db_name = self.last_used_db if self.last_used_db else None

        # Searches for the most relevant data entries
        if db_name in self.databases:
            self.last_used_db = db_name
            return db_name, self.databases[db_name].search(prompt, k)
        else:
            return None, []

    def reset_context(self):
        self.last_used_db = None

# CONVERSATION MANAGER #

class ConversationManager:
    def __init__(self):
        self.memory = ""
        self.current_exchange = {"user": "", "assistant": ""}
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def update(self, user_input: str, bot_response: str):
        self.current_exchange["user"] = user_input
        self.current_exchange["assistant"] = bot_response
        summary = self.generate_summary()
        self.memory = f"{self.memory} {summary}".strip()
        self.memory = self.memory[-500:]

    def generate_summary(self):
        prompt = f"""Summarize the following exchange in one concise sentence, as if you're explaining to Claude (the AI) what was discussed. Use "you" to refer to Claude/Impy, and "the user" for the human. Focus on the key points and any changes in topic:

User: {self.current_exchange['user']}
Assistant: {self.current_exchange['assistant']}

Summary:"""

        try:
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=50,
                system="You are a summarization assistant. In one sentence, provide a concise summary of the given exchange, written as if explaining to an AI what was just discussed.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return ""

    def get_context(self):
        return self.memory

# Claude time #

def get_claude_response(prompt, multi_db, conversation_manager, conversation_history):
    # Checks if it's a new conversation to reset the DB context
    if not conversation_history:
        multi_db.reset_context()

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
        messages = [
            *conversation_history,
            {"role": "user", "content": f"{formatted_info}\n\nConversation context:\n{context}\n\nUser's new question: {prompt}"}
        ]
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=messages
        )
        
        response = message.content[0].text
        conversation_manager.update(prompt, response)
        # finally, returns the response that Claude gives
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

def generate_streamed_response(response):
    paragraphs = response.split('\n')
    for paragraph in paragraphs:
        yield paragraph + '\n\n'
        time.sleep(0.1)  # Adjust the delay as needed
        words = paragraph.split()
        for word in words:
            yield word + ' '
            time.sleep(0.1)  # Adjust the delay as needed
        yield '\n\n'  # Add a new paragraph

def index(request):
    return render(request, 'chatbot_app/index.html')

@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            user_message = data.get('message', '')
            print(f"DEBUG: Received message: {user_message}")

            # Initialize user profile (this would be fetched from a real user session in a complete app)
            user_profile = {"year": "1", "course": "Materials"}  # Default for example

            # Initialize the database and conversation manager
            multi_db = MultiDB(user_profile)
            multi_db.load_databases('database')

            conversation_manager = ConversationManager()
            
            # Get Claude response
            response = get_claude_response(user_message, multi_db, conversation_manager)

            # Stream the response
            return StreamingHttpResponse(generate_streamed_response(response), content_type='text/plain')

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)

