#### Impy v3 ####



# Impy is an Imperial College Chatbot capable of assisting student and staff with administrative tasks.

# This mainly includes:

# -Providing information from the success guide (done)
# -Assignments based on the user's course and year (Currently have Materials Yr 1,2,3)
# -Personalised timetables (still work in progress) 

# Along with all the other capabilities that LLM's can provide, such as generating contextual responses.

# Impy is comprised of two main parts, its database searching capabilities and the Claude API.

# When the user asks the prompt, this python script looks at keywords to pull up the most relevant information.

# After that relevant information is obtained, that along with the system prompt, context of the conversation, and the
# original question is sent to the Claude API.

# Finally, the API returns a response.

# Look through all the comments that I've put across the script, I hope it's all clear.

# For any questions please contact me at sh5320@ic.ac.uk

# -Shawn

# Side note: All the DEBUG stuff is just for me to check the internals, and should be commented out for the final script.

# Libraries
import anthropic
import os
import json
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import re
from datetime import datetime
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from starlette.middleware.wsgi import WSGIMiddleware
from django.http import HttpResponse
from django.shortcuts import render

app = FastAPI()

# OBTAINING API KEY #
#
# Right now it's so that the user needs to export the key via the command:
#
# "export ANTHROPIC_API_KEY='x'"
#
# Where x is the key value.
#
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# DEFINING THE SYSTEM PROMPT #
#
# This is what makes Impy Impy. We're still in the early stages of refining the system prompt to be optimal.
# The Claude API is honestly pretty great at retaining character via the prompt, so you don't really need to
# repeat the same point over and over again to emphasise a point.
#
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

# USER PROFILING #
# This function is to ask the user for initial information on year and course.
# If we have access to the year and course automatically via login this is where we would change the variable.
def initialize_user_profile():
    year = input("What year are you in? (1/2/3/4): ")
    course = input("What is your course? (e.g., Materials): ")
    return {"year": year, "course": course}

### DATA ZONE ###

# This is the knowledge base of Impy. If you see the relevant files that are next to Impy3.py in the directory
# you will find multiple .JSON files. They're structured like a dictionary with multiple entries. This part of
# the script helps find the most relevant data entries regarding the user's prompts to eventually feed the 
# Claude API. 

# The first class, defining a single data entry.
class DataEntry:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata

#
# Here's an example of a data entry in the JSON file:
#
# {
#     "content": "Computing in-Class test (MATE50001 Mathematics and Computing 2)",
#     "metadata": {
#       "course": "MATE50001 Mathematics and Computing 2",
#       "due_date": "18/11/23",
#       "submission_method": "In-person",
#       "percentage": "15",
#       "assignment_type": "Test",
#       "Year": "2"
#     }
#   }
#
# The benefit of this data structure is that metadata is not strictly defined in these categories.
# The success guide entries for example has a URL section, category for student status, etc.
# Claude is able to contextualise these categories to give the most appropriate response.
#

# The second class takes all those data entries (the combined content and metadata) and vectorises them with 
# the TfidfVectorizer function. Once that is done, we use FAISS to index everything for extremely quick searching.
# 
class VectorDB:
    def __init__(self, name):
        self.name = name
        self.data = []
        self.vectorizer = TfidfVectorizer()
        self.index = None

    def add_entries(self, entries):
        self.data.extend(entries)
        self.build_index()

    # combines both content and metadata, and vectorises + indexes them
    def build_index(self):
        texts = [
            f"{entry.content} {' '.join(str(v) for v in entry.metadata.values())}"
            for entry in self.data
        ]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        # FAISS is Facebook's AI search and is extremely fast. We should not be worried about the amount of 
        # information we can store for now because it is super efficient.
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(tfidf_matrix.shape[1]))
        self.index.add_with_ids(tfidf_matrix.toarray().astype('float32'), np.array(range(len(texts))))

    # A search function that returns k number (currently 5) of the most relevant entries
    # It takes the query of the user, and vectorises it
    # After that, it compares with the indexed information and finds the top k matching results
    # Finally, it returns those k number of entries.
    def search(self, query, k=5):
        full_query = f"{query} {' '.join(str(v) for v in self.data[0].metadata.keys())}"
        query_vector = self.vectorizer.transform([full_query]).toarray().astype('float32')
        _, indices = self.index.search(query_vector, k)
        return [self.data[i] for i in indices[0]]

# The third class helps load the relevant databases based on the course and year into a contextual database
# for the user's instance.
class MultiDB:
    def __init__(self, user_profile):
        self.databases = {}
        self.user_profile = user_profile
        self.last_used_db = None

    def load_databases(self, base_path):
        # Ensure base_path is correct
        base_path = os.path.join(os.path.dirname(__file__), base_path)
        
        # Loads universal databases, information relevant to all Imperial student/staff.
        for file_name in os.listdir(base_path):
            if file_name.endswith('.json'):
                db_name = file_name[:-5]  # Remove .json extension
                file_path = os.path.join(base_path, file_name)
                self.add_database(db_name, file_path)

        # Load course-specific databases, based on the first asking of the user of course and year.
        course = self.user_profile['course']
        year = self.user_profile['year']
        course_path = os.path.join(base_path, course, f"Year {year}")
        
        if os.path.exists(course_path):
            for file_name in os.listdir(course_path):
                if file_name.endswith('.json'):
                    db_name = f"{course}_year{year}_{file_name[:-5]}"
                    file_path = os.path.join(course_path, file_name)
                    self.add_database(db_name, file_path)
        else:
            print(f"No specific data found for {course} Year {year}")

        # DEBUG: Print loaded databases for debugging
        print("Loaded databases:", list(self.databases.keys()))

    # Function to add the databases to the self.databases variable
    def add_database(self, name, file_path):
        db = VectorDB(name)
        with open(file_path, 'r') as f:
            data = json.load(f)
        entries = [DataEntry(item['content'], item.get('metadata', {})) for item in data]
        db.add_entries(entries)
        self.databases[name] = db

    # This function first analyzes based on keywords in the prompt which database should be used.
    # For example, if you talk about 'tests', the assignment database will be most relevant to you.
    # After that, it takes that database and does the search function that was in the VectorDB class.
    
    # There is an additional function that if there are no keywords in the following prompt, it will
    # still try to search the keywords of the query/prompt using the last used DB, to keep conversations 
    # flowing.

    def analyze_and_search(self, prompt, k=5):
        assignment_keywords = ['test', 'tests', 'coursework', 'exam', 'exams', 'assignment', 'assignments', 'deadline', 'due date']
        timetable_keywords = ['class', 'classes', 'lecture', 'lectures', 'seminar', 'lab', 'schedule', 'timetable']
        success_guide_keywords = ['success', 'tip', 'tips', 'guide', 'advice', 'help', 'study']

        prompt_lower = prompt.lower()
        
        if any(keyword in prompt_lower for keyword in assignment_keywords):
            db_name = f"{self.user_profile['course']}_year{self.user_profile['year']}_assignments"
        elif any(keyword in prompt_lower for keyword in timetable_keywords):
            db_name = f"{self.user_profile['course']}_year{self.user_profile['year']}_timetables"
        elif any(keyword in prompt_lower for keyword in success_guide_keywords):
            db_name = 'success_guide'
        else:
            db_name = self.last_used_db if self.last_used_db else None

        # Searches for the most relevant data entries
        if db_name in self.databases:
            self.last_used_db = db_name
            return db_name, self.databases[db_name].search(prompt, k)
        else:
            return None, []

    def reset_context(self):
        self.last_used_db = None

# CONVERSATION MANAGER #

# This is Impy's memory. Without it, each new prompt would be like the first interaction.
# In the first iteration I just gave Impy the previous 3 exchanges, but this proved to be a lot of text.
# My workaround was to use a baby version of Claude (Haiku 3) to analyse the conversation and summarise the conversation.
# This gets added to working memory that contains the summary of the ongoing conversation.
# This memory is then inserted to the final message we give to the main Claude API.

class ConversationManager:
    def __init__(self):
        self.memory = ""
        self.current_exchange = {"user": "", "assistant": ""}
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def update(self, user_input: str, bot_response: str):
        self.current_exchange["user"] = user_input
        self.current_exchange["assistant"] = bot_response
        summary = self.generate_summary()
        self.memory = f"{self.memory} {summary}".strip()
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

# CLAUDE TIME #

# This is where we actually call Claude. This contains all of the different bits and pieces perviously defined to make up Impy.

async def get_claude_response(prompt, multi_db, conversation_manager, is_new_conversation=False):
    # Checks if it's a new conversation to reset the DB context
    if is_new_conversation:
        multi_db.reset_context()

    # Helps find the relevant stuff based on the prompt
    result = multi_db.analyze_and_search(prompt)
    db_name, relevant_info = result[0], result[1]
    
    # Adds all the relevant info into a empty string called formatted_info
    formatted_info = ""
    if db_name and relevant_info:
        formatted_info = f"Relevant information from {db_name.upper()}:\n"
        for entry in relevant_info:
            formatted_info += f"- Content: {entry.content}\n"
            formatted_info += f"  Metadata: {json.dumps(entry.metadata, indent=2)}\n"
            for key, value in entry.metadata.items():
                if str(value).lower() in prompt.lower():
                    formatted_info += f"  Matched metadata: {key}: {value}\n"

    # Calls baby claude to give it its memory
    context = conversation_manager.get_context()

    # DEBUG: Shows formatted_info to see what got pulled up, and context to see how baby Claude summarised
    print(f"DEBUG - extracted info: {formatted_info}")
    print(f"DEBUG - Context used for API call: {context}")

    # Talks to Claude with the formatted info, the context/memory, and finally the prompt the user enters.
    # The main parameters are what model we are using, the max tokens used up, the system prompt we defined at the start, and the main message.
    #
    # If we were to change the API, this would be the place!!!
    #
    try:
        response = client.messages.create(
            # Currently using Claude's best model, efficient and powerful
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"{formatted_info}\n\nConversation context:\n{context}\n\nUser's new question: {prompt}"}
            ],
            stream=True #enable streaming mode (word-by-word response)
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

# Defines an endpoint to handle chat requests
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    user_profile = data.get("user_profile", {"year": "1", "course": "Materials"})
    multi_db = MultiDB(user_profile)
    multi_db.load_databases('database')
    conversation_manager = ConversationManager()

    return StreamingResponse(get_claude_response_stream(prompt, multi_db, conversation_manager), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# MAIN #
# This is where everything is loaded up, Claude, the databases, the conversation manager, etc.
def main():
    print("Welcome to Impy, your Imperial College London assistant!")
    # Firsts asks the user for year and course
    user_profile = initialize_user_profile()
    
    # Then loads up the relevant databases based on that info
    multi_db = MultiDB(user_profile)
    multi_db.load_databases('database')

    # Boots up the conversation manager for the memory
    conversation_manager = ConversationManager()

    print("Type 'exit' to end the conversation or 'new' to start a new conversation.")
    
    # Variable to figure out if the conversation is ongoing
    is_new_conversation = True

    # Will keep looping forever until a break
    while True:
        user_input = input("\nYou: ").strip()
        
        # You can write 'exit' to stop the conversation and the program
        if user_input.lower() == 'exit':
            print("Thank you for chatting with Impy. Goodbye!")
            break
        # You can write 'new' just to restart the conversation, does all those previous steps
        elif user_input.lower() == 'new':
            print("Starting a new conversation.")
            user_profile = initialize_user_profile()
            multi_db = MultiDB(user_profile)
            multi_db.load_databases('database')
            conversation_manager = ConversationManager()
            is_new_conversation = True
            continue
        
        # Calls on Claude and prints the conversation
        response = get_claude_response(user_input, multi_db, conversation_manager, is_new_conversation)
        print(f"\nImpy: {response}")
        
        # Notifies that this is now an ongoing conversation
        is_new_conversation = False


# The final step, just calls the main function to get things going
if __name__ == "__main__":
    main()


