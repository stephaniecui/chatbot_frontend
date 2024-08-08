import os
import json
from typing import List, Dict, Any
import anthropic
import openai
from datetime import datetime
import requests
import re

### AzureImpy 1.0 ###

# This is the python script that runs on Impy terminal/cmd.
# 
# You can easily change 

### CONFIGURATION ###

# Debug helps show what is fed to the LLM API
debug = True

# Select your api, "claude" or "gpt"
active_api = "gpt"

## All the environment variables, please run the .sh/bat file I sent!

# API Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # Currently using from OpenAI directly rather than Azure's one due to token constraints

# Model Configuration
CLAUDE_MAIN_MODEL = "claude-3-5-sonnet-20240620"
CLAUDE_SUMMARY_MODEL = "claude-3-haiku-20240307"
GPT_MAIN_MODEL = "gpt-4o"
GPT_SUMMARY_MODEL = "gpt-4o-mini"

# Azure AI Search Configuration
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME")

# Initialize API clients
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
openai.api_key = OPENAI_API_KEY


### SYSTEM PROMPT ###

SYSTEM_PROMPT = """Hello there! I'm Impy, the friendly Imperial College London Success Guide chatbot. My purpose is to be a helpful and approachable guide for students on the range of topics covered in the Imperial College London Success Guide. I speak in British English, no z's!
 
When you send me a message, it will be structured in three parts:
Relevant information pulled from the database to answer your query.
A brief summary of our conversation so far to provide context.
Your new question or prompt.
 
I'll use the database information to give you accurate, up-to-date answers. And the conversation history will help me maintain a natural flow and keep our chat on track.
 
If this is a brand new conversation, I'll greet you warmly and get us started. And even if the database doesn't have all the details, I'll do my best to provide helpful advice based on my knowledge of Imperial College. I may suggest reaching out to specific experts (e.g., professors, GTAs, lab technicians) in your department for more specialised information.
 
Whenever possible, I'll include relevant website links, especially if I'm unsure about something or the info isn't in the database. The main Imperial College site at https://www.imperial.ac.uk is a great general resource.
 
Now, if you seem a bit stressed or worried about something, I'll be sure to emphasise the available support resources and offer a kind, empathetic ear. My goal is to be a friendly, knowledgeable, and reassuring presence for the Imperial community.
 
So, how can I assist you today? I'm excited to put my capabilities to work and help make your Imperial experience as smooth and successful as possible. Fire away with your question or request, and let's get started!"""

### AZURE ###

# The Azure techstack accounts for storing the database and searching up relevant information.
# This replaces the DataEntry, VectorDB and MultiDB classes from previous versions of Impy.
# It analyses the "query" from the user which is semantically processed
# After that, a vector search is performed to the Azure database which has all the JSON files collated in it
# The top results are what get sent back to the LLM API (currently returns k=2 results)

def azure_search(query: str, k: int = 2) -> List[Dict[str, Any]]:
    # This is the URL that goes to my personal Azure account
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
        "select": "title, chunk" # Takes the title of the JSON file as well as the most relevant chunk
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        return response.json().get('value', [])
    else:
        print(f"Search API error: {response.status_code}, {response.text}")
        return []

# This formats the title of the JSON file and the top scoring chunk
def process_results(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"title": result.get('title', 'N/A'), "chunk": result.get('chunk', 'N/A')} for result in results]

# This formats it for the multiple k results
def format_for_impy(query: str, processed_results: List[Dict[str, str]]) -> str:
    formatted_info = f"Query: {query}\n\n"
    for result in processed_results:
        formatted_info += f"Title: {result['title']}\nChunk: {result['chunk']}\n\n"
    return formatted_info

# Function to call the Claude API
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

# Function to call the GPT API
def get_gpt_response(prompt: str, model: str, max_tokens: int = 1000) -> str:
    try:
        response = openai.chat.completions.create(
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

# Simple approximation to find out rough token usage amongst both APIs
def get_token_count(text: str) -> int:
    return len(text.split())

# General function that uses either Claude or GPT from the functions defined previously
# If we were to use a different API, we can easily add it as another conditional
def get_ai_response(prompt: str, conversation_manager, log_file_path: str, is_new_conversation: bool = False) -> str:
    search_results = azure_search(prompt)
    processed_results = process_results(search_results)
    formatted_info = format_for_impy(prompt, processed_results)
    
    context = conversation_manager.get_context()
    
    full_prompt = f"{formatted_info}\n\nConversation context:\n{context}\n\nUser's new question: {prompt}"

    # Debug to see what was sent to the API 
    if debug:
            print(f"\nDEBUG - Formatted info from search:\n{formatted_info}")
            print(f"\nDEBUG - Conversation context:\n{context}")
            print(f"\nDEBUG - You:\n{full_prompt}")
        
    # Which API is called
    if active_api == 'claude':
        response = get_claude_response(full_prompt, CLAUDE_MAIN_MODEL)
        prompt_tokens = get_token_count(full_prompt)
        response_tokens = get_token_count(response)
    else:  # GPT
        response = get_gpt_response(full_prompt, GPT_MAIN_MODEL)
        prompt_tokens = get_token_count(full_prompt)
        response_tokens = get_token_count(response)
    
    token_usage = {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens
    }
    
    log_interaction(log_file_path, prompt, response, token_usage)
    
    conversation_manager.update(prompt, response)
    return response

### Conversation Manager ###

# This class helps give Impy a memory, by summarising past interactions using a secondary call of the API
# It uses the cheaper version of the API (Haiku for Claude, 4o-mini for GPT)
class ConversationManager:
    def __init__(self, max_memory_length: int = 1000):
        self.memory = ""
        self.current_exchange = {"user": "", "assistant": ""}
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

### Logger ###

# This just helps record your entire conversation with Impy before you exit, including token usage as well
def create_log_file() -> str:
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    date_dir = os.path.join(logs_dir, datetime.now().strftime("%d_%m_%y"))
    os.makedirs(date_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%H_%M_%S")
    log_file_name = f"{timestamp}.txt"
    return os.path.join(date_dir, log_file_name)

def log_interaction(log_file_path: str, user_input: str, assistant_response: str, token_usage: Dict[str, int]):
    with open(log_file_path, "a") as log_file:
        log_file.write(f"User: {user_input}\n")
        log_file.write(f"Assistant: {assistant_response}\n")
        log_file.write(f"Token usage: {json.dumps(token_usage)}\n")
        log_file.write("\n")

### Main ###

# The main loop which helps run the chatbot on terminal.

def main():
    print(f"Welcome to Impy, your Imperial College London assistant! (Using {active_api.upper()} API)")

    # Tells you which API you are using
    if active_api == 'claude':
        print(f"Main model: {CLAUDE_MAIN_MODEL}, Summary model: {CLAUDE_SUMMARY_MODEL}")
    else:
        print(f"Main model: {GPT_MAIN_MODEL}, Summary model: {GPT_SUMMARY_MODEL}")
    
    log_file_path = create_log_file()
    conversation_manager = ConversationManager()

    print("Type 'exit' to end the conversation or 'new' to start a new conversation.")
    
    is_new_conversation = True
    
    # Main Loop
    while True:
        # Records the user's input/prompt/query
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'exit':
            print("Thank you for chatting with Impy. Goodbye!")
            break
        elif user_input.lower() == 'new':
            print("Starting a new conversation.")
            conversation_manager = ConversationManager()
            is_new_conversation = True
            continue
        

        # What Impy responds with
        response = get_ai_response(user_input, conversation_manager, log_file_path, is_new_conversation)
        print(f"\nImpy: {response}")
        
        is_new_conversation = False

if __name__ == "__main__":
    main()