import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import render
from django.http import JsonResponse
from .models import ChatMessage
from .api_utils import get_claude_response
from django.views.decorators.csrf import csrf_exempt
import json

def index(request):
    return render(request, 'chatbot_app/index.html')

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

# Define the system prompt
SYSTEM_PROMPT = """You are the Imperial College London Chatbot, "Impy", designed to aid Imperial College students and staff on administrative and education matters. 
Your responses should be helpful, friendly, and tailored to the Imperial College community. 
Use the provided Imperial College information to answer questions accurately. If you're unsure or the information isn't available, please indicate that and suggest where the user might find accurate information."""

# Function to scrape and process web content
def scrape_imperial_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    for elem in soup(['header', 'nav', 'footer']):
        elem.decompose()
    
    main_content = soup.find('main') or soup.find(id='main-content') or soup.find(class_='main-content')
    
    if main_content:
        text = main_content.get_text(separator=' ', strip=True)
    else:
        text = soup.get_text(separator=' ', strip=True)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text[:2000]

# Scrape important pages
pages = {
    'research': scrape_imperial_website('https://www.imperial.ac.uk/research-and-innovation/'),
    'study': scrape_imperial_website('https://www.imperial.ac.uk/study/'),
    'students': scrape_imperial_website('https://www.imperial.ac.uk/students/'),
}

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(pages.values())

def get_relevant_info(query, top_n=2):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    relevant_info = ""
    for idx in top_indices:
        page_name = list(pages.keys())[idx]
        relevant_info += f"Information from {page_name} page:\n"
        relevant_info += pages[list(pages.keys())[idx]] + "\n\n"
    
    return relevant_info

class ConversationManager:
    def __init__(self):
        self.context_summary = ""
        self.recent_exchanges = []

    def update(self, user_input, bot_response):
        self.recent_exchanges.append(("user", user_input))
        self.recent_exchanges.append(("assistant", bot_response))

        self.recent_exchanges = self.recent_exchanges[-6:]

        self.context_summary += f"\nUser asked about: {user_input}\nKey points from response: {bot_response[:100]}..."

        if len(self.context_summary) > 1000:
            self.context_summary = self.context_summary[-1000:]

    def get_context(self):
        context = f"Context summary: {self.context_summary}\n\nRecent exchanges:\n"
        for role, content in self.recent_exchanges:
            context += f"{role.capitalize()}: {content}\n"
        return context

conversation_manager = ConversationManager()

def get_claude_response(prompt):
    relevant_info = get_relevant_info(prompt)
    context = conversation_manager.get_context()

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Relevant Imperial College information:\n{relevant_info}\n\nConversation context:\n{context}\n\nUser's new question: {prompt}"}
            ]
        )
        response = message.content[0].text
        
        conversation_manager.update(prompt, response)
        
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Scrape important pages
pages = {
    'research': scrape_imperial_website('https://www.imperial.ac.uk/research-and-innovation/'),
    'study': scrape_imperial_website('https://www.imperial.ac.uk/study/'),
    'students': scrape_imperial_website('https://www.imperial.ac.uk/students/'),
}
