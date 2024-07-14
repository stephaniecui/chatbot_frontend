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
