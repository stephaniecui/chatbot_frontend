# chatbot_app/claude_api.py

import anthropic
import os

# Initialize the client with your API key
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Define the system prompt
SYSTEM_PROMPT = """You are the Imperial College London Chatbot, "Impy", designed to aid Imperial College students and staff on administrative and education matters. 
Your responses should be helpful, friendly, and tailored to the Imperial College community. 
Use the provided Imperial College information to answer questions accurately. If you're unsure or the information isn't available, please indicate that and suggest where the user might find accurate information."""

class ConversationManager:
    def __init__(self):
        self.context_summary = ""
        self.recent_exchanges = []

    def update(self, user_input, bot_response):
        # Add the new exchange
        self.recent_exchanges.append(("user", user_input))
        self.recent_exchanges.append(("assistant", bot_response))

        # Keep only the last 3 exchanges
        self.recent_exchanges = self.recent_exchanges[-6:]

        # Update context summary
        self.context_summary += f"\nUser asked about: {user_input}\nKey points from response: {bot_response[:100]}..."

        # Trim context summary if it gets too long
        if len(self.context_summary) > 1000:
            self.context_summary = self.context_summary[-1000:]

    def get_context(self):
        context = f"Context summary: {self.context_summary}\n\nRecent exchanges:\n"
        for role, content in self.recent_exchanges:
            context += f"{role.capitalize()}: {content}\n"
        return context

conversation_manager = ConversationManager()

def get_claude_response(prompt, relevant_info, context):
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
        
        # Update conversation manager
        conversation_manager.update(prompt, response)
        
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"
