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
import re

# Initialize the client with your API key
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Define the system prompt
SYSTEM_PROMPT = """You are the Imperial College London Chatbot, "Impy", designed to assist students and staff (the user) with various queries including but not limited to the Success Guide, timetables, and assignment due dates.
Your responses should be helpful, friendly, and tailored to the Imperial College community.
You will be given content extracted from a database based on the user's questions. 
If this is not provided, you don't need to say you don't have specific information from the database, just advise them reasonably within context.
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
        prompt = f"""Summarize the following exchange in one concise sentences, as if you're explaining to Claude (the AI) what was discussed. Use "you" to refer to Claude/Impy, and "the user" for the human. Focus on the key points and any changes in topic:

User: {self.current_exchange['user']}
Assistant: {self.current_exchange['assistant']}

Summary:"""

        try:
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=50,  # Increased slightly to allow for more natural language
                system="You are a summarization assistant. Provide a concise summary of the given exchange, written as if explaining to an AI what was just discussed.",
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
# Usage in the main loop
conversation_manager = ConversationManager()

### DATA ZONE ####
multi_db = MultiDB()
multi_db.add_database('success_guide', os.path.join(settings.BASE_DIR, 'database/success_guide.json'))
multi_db.add_database('timetables', os.path.join(settings.BASE_DIR, 'database/timetables.json'))
multi_db.add_database('assignments', os.path.join(settings.BASE_DIR, 'database/assignments.json'))
multi_db.add_database('glossary', os.path.join(settings.BASE_DIR, 'database/glossary.json'))

### Experimental zone

class GlossaryManager:
    def __init__(self, glossary_file):
        with open(glossary_file, 'r') as f:
            self.glossary = json.load(f)
        self.term_map = {self.normalize_term(entry['metadata']['term']): entry for entry in self.glossary}
    
    def normalize_term(self, term):
        return re.sub(r'[^\w\s]', '', term.lower())
    
    def get_definition(self, query):
        words = self.normalize_term(query).split()
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                sub_query = ' '.join(words[i:j])
                if sub_query in self.term_map:
                    entry = self.term_map[sub_query]
                    return entry['metadata']['term'], entry['content']
        return None, None

def is_definition_query(query):
    definition_indicators = [
        r'\b(what|define|explain|tell me about|meaning of)\b',
        r'\bis\b.*\?',
        r'\bmean[s]?\b',
        r'\bdefinition\b'
    ]
    
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in definition_indicators)

def process_user_input(user_input):
    if not is_definition_query(user_input):
        return None

    # Extract the potential term from the query
    pattern = r'(?:what|define|explain|tell me about|meaning of).*?([\w\s]+)(?:\?)?$'
    match = re.search(pattern, user_input.lower())
    
    if match:
        query = match.group(1).strip()
        term, definition = glossary_manager.get_definition(query)
        if definition:
            return f"The term '{term}' is defined as: {definition}"
    
    return None

# Initialize the GlossaryManager
glossary_manager = GlossaryManager(os.path.join(settings.BASE_DIR, 'database/glossary.json'))

### Claude time

def get_claude_response(prompt: str) -> str:

    #Glossary function to avoid token
    glossary_response = process_user_input(prompt)
    if glossary_response:
        return glossary_response

    relevant_info = multi_db.search_all(prompt)
    #Context
    context = conversation_manager.get_context()
    #print(f"DEBUG - Context used for API call: {context}")

    # Format the relevant_info for Claude
    formatted_info = "Relevant information:\n


