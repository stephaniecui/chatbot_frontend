# chatbot_app/views.py

import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import render
from django.http import JsonResponse
from .models import ChatMessage
from .claude_api import get_claude_response, conversation_manager
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

            # Scrape important pages
            pages = {
                'research': scrape_imperial_website('https://www.imperial.ac.uk/research-and-innovation/'),
                'study': scrape_imperial_website('https://www.imperial.ac.uk/study/'),
                'students': scrape_imperial_website('https://www.imperial.ac.uk/students/'),
            }

            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(pages.values())

            relevant_info = get_relevant_info(user_message, pages, vectorizer, tfidf_matrix)
            context = conversation_manager.get_context()

            bot_response = get_claude_response(user_message, relevant_info, context)
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

# Function to get relevant info
def get_relevant_info(query, pages, vectorizer, tfidf_matrix, top_n=2):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    relevant_info = ""
    for idx in top_indices:
        page_name = list(pages.keys())[idx]
        relevant_info += f"Information from {page_name} page:\n"
        relevant_info += pages[list(pages.keys())[idx]] + "\n\n"
    
    return relevant_info

