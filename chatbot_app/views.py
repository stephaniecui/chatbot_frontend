import anthropic
import os
import json
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import ChatMessage
from django.conf import settings

# Initialize the client with your API key
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Define the system prompt
SYSTEM_PROMPT = """You are the Imperial College London Chatbot, "Impy", designed to assist students and staff with various queries including but not limited to the Success Guide, timetables, and assignment due dates.
Your responses should be helpful, friendly, and tailored to the Imperial College community. 
Use the provided information to answer questions accurately. If you're unsure or the information isn't available, please indicate that and suggest where the user might find accurate information."""

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
        texts = [entry.content for entry in self.data]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(tfidf_matrix.shape[1]))
        self.index.add_with_ids(tfidf_matrix.toarray().astype('float32'), np.array(range(len(texts))))

    def search(self, query: str, k: int = 3) -> List[DataEntry]:
        query_vector = self.vectorizer.transform([query]).toarray().astype('float32')
        _, indices = self.index.search(query_vector, k)
        return [self.data[i] for i in indices[0]]

class MultiDB:
    def __init__(self):
        self.databases: Dict[str, VectorDB] = {}

    def add_database(self, name: str, data_path: str):
        db = VectorDB(name)
        with open(data_path, 'r') as f:
            data = json.load(f)
        entries = [DataEntry(item['content'], item.get('metadata', {})) for item in data]
        db.add_entries(entries)
        self.databases[name] = db

    def search_all(self, query: str, k: int = 3) -> Dict[str, List[DataEntry]]:
        results = {}
        for name, db in self.databases.items():
            results[name] = db.search(query, k)
        return results

class ConversationManager:
    def __init__(self):
        self.context_summary = ""
        self.recent_exchanges = []

    def update(self, user_input, bot_response):
        self.recent_exchanges.append(("user", user_input))
        self.recent_exchanges.append(("assistant", bot_response))
        self.recent_exchanges = self.recent_exchanges[-6:]
        self.context_summary = self.summarize_text(bot_response)

    def summarize_text(self, text: str, num_sentences: int = 2) -> str:
        sentences = text.split('. ')
        return '. '.join(sentences[:num_sentences]) + ('.' if len(sentences) > num_sentences else '')

    def get_context(self) -> str:
        context = f"Context summary: {self.context_summary}\n\nRecent exchanges:\n"
        for role, content in self.recent_exchanges:
            context += f"{role.capitalize()}: {content}\n"
        return context

multi_db = MultiDB()
multi_db.add_database('success_guide', os.path.join(settings.BASE_DIR, 'database/success_guide.json'))
multi_db.add_database('timetables', os.path.join(settings.BASE_DIR, 'database/timetables.json'))
multi_db.add_database('assignments', os.path.join(settings.BASE_DIR, 'database/assignments.json'))

conversation_manager = ConversationManager()

def get_claude_response(prompt: str) -> str:
    relevant_info = multi_db.search_all(prompt)
    context = conversation_manager.get_context()

    formatted_info = "Relevant information:\n"
    for db_name, entries in relevant_info.items():
        formatted_info += f"\n{db_name.upper()}:\n"
        for entry in entries:
            formatted_info += f"- Content: {entry.content}\n"
            formatted_info += f"  Metadata: {json.dumps(entry.metadata, indent=2)}\n"

    try:
        message = client.messages.create(
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

def index(request):
    return render(request, 'chatbot_app/index.html', {"welcome_message": "Welcome to Impy, your Imperial College London assistant! Type 'exit' to end the conversation."})

@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            user_message = data.get('message', '')
            print(f"DEBUG: Received message: {user_message}")

            bot_response = get_claude_response(user_message)
            bot_response = bot_response.replace("\n", "<br>")

            ChatMessage.objects.create(user_message=user_message, bot_response=bot_response)
            print(f"DEBUG: Sending response: {bot_response}")
            return JsonResponse({'response': bot_response})
        except json.JSONDecodeError:
            print("DEBUG: Invalid JSON")
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        print("DEBUG: Invalid request method")
        return JsonResponse({'error': 'Invalid request method'}, status=400)

