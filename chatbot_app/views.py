import anthropic
import os
import json
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import re
from datetime import datetime
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from starlette.middleware.wsgi import WSGIMiddleware
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.conf import settings

app = FastAPI()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
        self.memory = ""
        self.current_exchange = {"user": "", "assistant": ""}
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def update(self, user_input: str, bot_response: str):
        self.current_exchange["user"] = user_input
        self.current_exchange["assistant"] = bot_response

        # Generate a summary of the current exchange
        summary = self.generate_summary()

        # Update the memory with the new summary
        self.memory = f"{self.memory} {summary}".strip()

        # Trim memory if it gets too long (adjust the number as needed)
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

multi_db = MultiDB()
multi_db.add_database('success_guide', os.path.join(settings.BASE_DIR, 'database/success_guide.json'))
multi_db.add_database('timetables', os.path.join(settings.BASE_DIR, 'database/timetables.json'))
multi_db.add_database('assignments', os.path.join(settings.BASE_DIR, 'database/assignments.json'))
multi_db.add_database('glossary', os.path.join(settings.BASE_DIR, 'database/glossary.json'))

conversation_manager = ConversationManager()

async def get_claude_response(prompt: str, multi_db, conversation_manager, is_new_conversation=False):
    if is_new_conversation:
        multi_db.reset_context()

    result = multi_db.analyze_and_search(prompt)
    db_name, relevant_info = result[0], result[1]
    
    formatted_info = ""
    if db_name and relevant_info:
        formatted_info = f"Relevant information from {db_name.upper()}:\n"
        for entry in relevant_info:
            formatted_info += f"- Content: {entry.content}\n"
            formatted_info += f"  Metadata: {json.dumps(entry.metadata, indent=2)}\n"
            for key, value in entry.metadata.items():
                if str(value).lower() in prompt.lower():
                    formatted_info += f"  Matched metadata: {key}: {value}\n"

    context = conversation_manager.get_context()

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"{formatted_info}\n\nConversation context:\n{context}\n\nUser's new question: {prompt}"}
            ],
            stream=True  # Enable streaming mode
        )

        async for event in response:
            if 'choices' in event and event['choices']:
                text = event['choices'][0]['text']
                for word in text.split():
                    yield word + " "
                    await asyncio.sleep(0.1)  # Adjust the speed of word streaming

        conversation_manager.update(prompt, response)

    except Exception as e:
        yield f"An error occurred: {str(e)}"

@csrf_exempt
@app.post("/chatbot/chat/")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    user_profile = data.get("user_profile", {"year": "1", "course": "Materials"})
    multi_db = MultiDB(user_profile)
    multi_db.add_database('success_guide', os.path.join(settings.BASE_DIR, 'database/success_guide.json'))
    multi_db.add_database('timetables', os.path.join(settings.BASE_DIR, 'database/timetables.json'))
    multi_db.add_database('assignments', os.path.join(settings.BASE_DIR, 'database/assignments.json'))
    multi_db.add_database('glossary', os.path.join(settings.BASE_DIR, 'database/glossary.json'))
    conversation_manager = ConversationManager()

    return StreamingResponse(get_claude_response(prompt, multi_db, conversation_manager), media_type="text/plain")

# Simple Django view for the root URL
def index(request):
    return render(request, 'chatbot_app/index.html')

@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            user_message = data.get('message', '')
            print(f"DEBUG: Received message: {user_message}")

            response = get_claude_response(user_message, multi_db, conversation_manager)
            return JsonResponse({'response': response})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
        
# Integrate FastAPI with Django using WSGIMiddleware
fastapi_app = WSGIMiddleware(app)
