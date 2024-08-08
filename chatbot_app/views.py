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
import requests
import anthropic
import openai  # Ensure OpenAI is imported
import nltk
from nltk.corpus import stopwords
import markdown2

# Select your API, "claude" or "gpt"
active_api = "gpt"

# API Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Model Configuration
CLAUDE_MAIN_MODEL = "claude-3-5-sonnet-20240620"
CLAUDE_SUMMARY_MODEL = "claude-3-haiku-20240307"
GPT_MAIN_MODEL = "gpt-4o"
GPT_SUMMARY_MODEL = "gpt-4o-mini"

# Azure AI Search Configuration
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME")

# Ensure environment variables are set
if not all([ANTHROPIC_API_KEY, OPENAI_API_KEY, AZURE_SEARCH_KEY, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME]):
    raise ValueError("One or more required environment variables are missing")

# Initialize API clients
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
openai.api_key = OPENAI_API_KEY  # Ensure this is correctly set

SYSTEM_PROMPT = """Hello there! I'm Impy, the friendly Imperial College London Success Guide chatbot. My purpose is to be a helpful and approachable guide for students on the range of topics covered in the Imperial College London Success Guide. I speak in British English, no z's!

When you send me a message, I'll provide a response structured to directly address your specific question or query. I’ll present the information in a clear, easy-to-read format using bullet points, numbered lists, concise paragraphs or other formatting as appropriate. I will avoid using the Markdown-style syntax.

First, I'll draw on relevant information from the database to directly address your specific question or query. Then, if a brief summary of our conversation so far would provide helpful context, I'll include that. But if the flow of the discussion is clear, I won't interrupt it with an unnecessary recap. Finally, I'll conclude by prompting you for your new question. This three-part structure will ensure my responses are tailored, informative and coherent. I won't introduce unrelated examples or tangents, as that could be confusing. Instead, I'll keep my focus squarely on being directly responsive to what you've asked.

If this is a brand new conversation, I'll greet you warmly and prompt you for your question or request. I won't make assumptions about your level of study. When the database doesn't have all the details, I'll do my best to provide helpful advice based on my knowledge of Imperial College London. I may suggest reaching out to specific experts (e.g., professors, GTAs, lab technicians) in your department for more specialised information.

Whenever possible, I'll include relevant website links, especially if I'm unsure about something or the info isn't in the database. The main Imperial College site at https://www.imperial.ac.uk is a great general resource.

Now, if you seem a bit stressed or worried about something, I'll be sure to emphasise the available support resources and offer a kind, empathetic ear. My goal is to be a friendly, knowledgeable, and reassuring presence for the Imperial community.

So, how can I assist you today? I'm excited to put my capabilities to work and help make your Imperial experience as smooth and successful as possible. Fire away with your question or request, and let's get started!"""

# The Azure tech stack accounts for storing the database and searching up relevant information.
def azure_search(query: str, k: int = 2) -> List[Dict[str, Any]]:
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2023-07-01-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_KEY
    }
    body = {
        "search": query,
        "queryType": "semantic",
        "semanticConfiguration": "vector-1722855681222-semantic-configuration",
        "top": k,
        "queryLanguage": "en-gb",
        "select": "title, chunk"
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        return response.json().get('value', [])
    else:
        print(f"Search API error: {response.status_code}, {response.text}")
        return []

def process_results(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"title": result.get('title', 'N/A'), "chunk": result.get('chunk', 'N/A')} for result in results]

def format_for_impy(query: str, processed_results: List[Dict[str, str]]) -> str:
    formatted_info = f"Query: {query}\n\n"
    for result in processed_results:
        formatted_info += f"Title: {result['title']}\nChunk: {result['chunk']}\n\n"
    return formatted_info

def get_claude_response(prompt: str, model: str, max_tokens: int = 1000) -> str:
    try:
        message = anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"An error occurred with the Claude API: {str(e)}"

def get_gpt_response(prompt: str, model: str, max_tokens: int = 1000) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the GPT API: {str(e)}"

def get_ai_response(prompt: str, conversation_manager, is_new_conversation: bool = False, is_regenerate: bool = False) -> str:
    search_results = azure_search(prompt)
    processed_results = process_results(search_results)
    formatted_info = format_for_impy(prompt, processed_results)
    
    context = conversation_manager.get_context()
    
    regenerate_instruction = """
        The user has requested a different answer to their previous question. 
        They were not satisfied with the initial response. 
        Please generate a new, alternative answer to the same question. 
        Provide different information, a new perspective, or elaborate on aspects 
        not covered in the initial response. Ensure this new answer is distinct 
        from the previous one while still accurately addressing the user's question.
        """ if is_regenerate else ""
    
    full_prompt = f"{formatted_info}\n\nConversation context:\n{context}\n\nUser's new question: {prompt}\n\n{regenerate_instruction}"
    
    if active_api == 'claude':
        response = get_claude_response(full_prompt, CLAUDE_MAIN_MODEL)
    else:  # GPT
        response = get_gpt_response(full_prompt, GPT_MAIN_MODEL)
    
    conversation_manager.update(prompt, response)
    
     # Format hyperlinks
    response_with_links = format_hyperlinks(response)
    # Convert markdown to HTML
    response_with_html = markdown2.markdown(response_with_links)
    return response_with_html

class ConversationManager:
    def __init__(self, memory: str = "", current_exchange: Dict[str, str] = None, max_memory_length: int = 1000):
        self.memory = memory
        self.current_exchange = current_exchange if current_exchange else {"user": "", "assistant": ""}
        self.max_memory_length = max_memory_length

    def update(self, user_input: str, bot_response: str):
        self.current_exchange["user"] = user_input
        self.current_exchange["assistant"] = bot_response
        summary = self.generate_summary()
        self.memory = self.add_to_memory(summary)

    def generate_summary(self) -> str:
        prompt = f"""I summarize the following exchange in one concise sentence, as if I'm explaining to myself previously on what happened. I use "I" to refer to myself, and "the user" for the user. I focus on the key points and any changes in topic:

User: {self.current_exchange['user']}
Assistant: {self.current_exchange['assistant']}

Summary:"""

        if active_api == 'claude':
            summary = get_claude_response(prompt, CLAUDE_SUMMARY_MODEL, max_tokens=100)
        else:  # GPT
            summary = get_gpt_response(prompt, GPT_SUMMARY_MODEL, max_tokens=100)

        return summary.strip()

    def add_to_memory(self, new_summary: str) -> str:
        if len(self.memory) + len(new_summary) <= self.max_memory_length:
            return f"{self.memory} {new_summary}".strip()
        else:
            sentences = re.split(r'(?<=[.!?])\s+', self.memory)
            while len(' '.join(sentences)) + len(new_summary) > self.max_memory_length:
                sentences.pop(0)
            return f"{' '.join(sentences)} {new_summary}".strip()

    def get_context(self) -> str:
        return self.memory

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory": self.memory,
            "current_exchange": self.current_exchange,
            "max_memory_length": self.max_memory_length
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationManager':
        return cls(
            memory=data.get("memory", ""),
            current_exchange=data.get("current_exchange", {"user": "", "assistant": ""}),
            max_memory_length=data.get("max_memory_length", 1000)
        )

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
    request.session.flush()
    return render(request, 'chatbot_app/index.html')

@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_input = data.get('message')
            is_new_conversation = request.session.get('is_new_conversation', False)

            # Initialize session if it's the first request
            if 'initialized' not in request.session:
                request.session['initialized'] = True
                request.session['is_new_conversation'] = True
                greeting = "Hello there! I'm Impy, your friendly Imperial College London Success Guide chatbot. How can I assist you today?"
                return StreamingHttpResponse(generate_streamed_response(greeting), content_type='text/plain')

            if not user_input:
                return JsonResponse({'error': 'No message provided'}, status=400)
            
            conversation_manager_data = request.session.get('conversation_manager', None)
            if conversation_manager_data:
                conversation_manager = ConversationManager.from_dict(conversation_manager_data)
            else:
                conversation_manager = ConversationManager()
                request.session['conversation_manager'] = conversation_manager.to_dict()

            is_regenerate = data.get('is_regenerate', False)
            response = get_ai_response(user_input, conversation_manager, is_new_conversation, is_regenerate)
            
            # Update session information
            request.session['is_new_conversation'] = False
            request.session['conversation_manager'] = conversation_manager.to_dict()

            return StreamingHttpResponse(generate_streamed_response(response), content_type='text/plain')

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
