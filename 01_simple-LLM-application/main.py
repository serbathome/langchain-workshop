

"""
Simple LLM Application with Azure OpenAI

This example demonstrates the fundamentals of working with Large Language Models using Azure OpenAI.
It covers:
1. Basic Azure OpenAI setup and configuration
2. Direct message-based interactions with the LLM
3. System and user message handling
4. Prompt templates for dynamic content generation
5. Translation use cases with different target languages

This is the foundation for understanding how to communicate with LLMs before moving
to more complex patterns like memory management, agents, and RAG systems.
"""

# Import necessary libraries
import os  # Operating system interface for environment variables
from langchain_openai import AzureChatOpenAI  # Azure OpenAI integration
from langchain_core.messages import HumanMessage, SystemMessage  # Message types
from langchain_core.prompts import ChatPromptTemplate  # Template for dynamic prompts
from dotenv import load_dotenv  # Environment variable loader

# Load environment variables from .env file
# This loads Azure OpenAI credentials and configuration
load_dotenv()

# ============================================================================
# STEP 1: Initialize Azure OpenAI Model
# ============================================================================

# Create an instance of AzureChatOpenAI with environment variables
# This establishes connection to your Azure OpenAI resource
model = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),        # Your Azure OpenAI endpoint URL
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"), # Name of your deployed model
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"), # API version for compatibility
    openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),         # Your API key for authentication
)

# ============================================================================
# STEP 2: Direct Message-Based Interaction
# ============================================================================

# Define the messages to be sent to the model
# SystemMessage: Provides instructions and context to the model
# HumanMessage: Represents the user's input or question
messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

print("🔄 Example 1: Direct message translation (English → Italian)")
print(f"System instruction: {messages[0].content}")
print(f"User input: {messages[1].content}")

# Invoke the model with the messages and print the response
# The model processes both the system instruction and user input
result = model.invoke(messages)
print(f"🤖 AI Response: {result.content}")
print("-" * 60)

# ============================================================================
# STEP 3: Using Prompt Templates for Dynamic Content
# ============================================================================

# Alternatively, using a prompt template for more flexible interactions
# Prompt templates allow you to create reusable patterns with variable substitution
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

print("🔄 Example 2: Template-based translation (English → German)")
print(f"Template: {system_template}")

# Create a specific prompt by filling in the template variables
prompt = prompt_template.invoke({"language": "German", "text": "hi!"})
print(f"Generated prompt: System='{prompt.messages[0].content}', User='{prompt.messages[1].content}'")

# Invoke the model with the generated prompt and print the response
response = model.invoke(prompt)
print(f"🤖 AI Response: {response.content}")
print("-" * 60)

# ============================================================================
# STEP 4: Interactive User Input (Exercise)
# ============================================================================

print("🎯 Exercise: Interactive Translation")
print("This demonstrates how to create interactive applications with user input")

user_text = input("Please enter the text to translate: ")
user_language = input("Please enter the target language: ")

# Create a prompt using the template with user-provided values
user_prompt = prompt_template.invoke({"language": user_language, "text": user_text})

print(f"\n🔄 Translating '{user_text}' to {user_language}...")

# Get translation from the model
translation_response = model.invoke(user_prompt)
print(f"🤖 Translation: {translation_response.content}")

# ============================================================================
# Key Concepts Demonstrated:
# ============================================================================
"""
This simple LLM application demonstrates several fundamental concepts:

1. AZURE OPENAI SETUP:
   - Environment variable configuration for secure credential management
   - Model initialization with proper Azure endpoints and API versions
   - Authentication using API keys

2. MESSAGE TYPES:
   - SystemMessage: Provides context, instructions, and personality to the AI
   - HumanMessage: Represents user input and questions
   - Message ordering affects how the AI interprets the conversation

3. PROMPT TEMPLATES:
   - Reusable patterns with variable substitution using {variable} syntax
   - Separation of template structure from dynamic content
   - More maintainable than hardcoded strings

4. MODEL INVOCATION:
   - Direct model.invoke() calls for immediate responses
   - Synchronous communication pattern
   - Response objects containing generated content

5. INTERACTIVE APPLICATIONS:
   - User input collection using input() function
   - Dynamic prompt generation based on user data
   - Real-time interaction with AI models

Benefits of this approach:
- ✅ Simple and direct communication with LLMs
- ✅ Template-based flexibility for different use cases
- ✅ Foundation for more complex applications
- ✅ Easy to understand and modify

Next Steps:
- Add conversation memory (see 02_chatbot-application)
- Implement tool usage and agents (see 03_simple-agent-application)
- Build knowledge-based systems with RAG (see 04_simple-rag-application)
"""

