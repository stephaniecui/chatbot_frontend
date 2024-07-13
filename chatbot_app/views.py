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
        user_message = request.POST.get('message')
        if user_message:
            print(f"DEBUG: Received message: {user_message}")
            bot_response = get_claude_response(user_message)
            ChatMessage.objects.create(user_message=user_message, bot_response=bot_response)
            print(f"DEBUG: Sending response: {bot_response}")
            return JsonResponse({'response': bot_response})
        else:
            print("DEBUG: No message received")
            return JsonResponse({'response': 'No message received'}, status=400)
    else:
        return JsonResponse({'response': 'Invalid request'}, status=400)


