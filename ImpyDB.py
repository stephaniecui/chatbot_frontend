import anthropic
import os
import json
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import re
from datetime import datetime

# Initialize the client with your API key
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Define the system prompt
SYSTEM_PROMPT = """You are the Imperial College London Chatbot, "Impy", designed to assist students and staff (the user) with various queries including but not limited to the Success Guide, timetables, and assignment due dates.
Your responses should be helpful, friendly, and tailored to the Imperial College community.
You will be given content extracted from a full database based on the user's questions. You technically have all the information but it's dependent on what the user asks contextually.
You should not tell the user any of the behind the scenes you are doing, and just tell the user what you know is from a subset selection of data. 
Give the relevant URLs when you can, especially if you're unsure or the information isn't available."""

### Database

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

### Conversation logic

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

### DATA ZONE ####

multi_db = MultiDB()
multi_db.add_database('success_guide', 'database/new_success_guide.json')
multi_db.add_database('timetables', 'database/timetables.json')
multi_db.add_database('assignments', 'database/assignments.json')
multi_db.add_database('glossary', 'database/glossary.json')



conversation_manager = ConversationManager()

### Experimental zone


### Claude time

def get_claude_response(prompt: str) -> str:
    relevant_info = multi_db.search_all(prompt)
    context = conversation_manager.get_context()

    # Format the relevant_info for Claude
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

def main():
    print("Welcome to Impy, your Imperial College London assistant!")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            print("Thank you for chatting with Impy. Goodbye!")
            break
        
        response = get_claude_response(user_input)
        print(f"\nImpy: {response}")

if __name__ == "__main__":
    main()