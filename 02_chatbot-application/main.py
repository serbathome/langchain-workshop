
"""
Interactive Chatbot Application with Memory and Streaming

This example demonstrates how to build a conversational AI chatbot that:
1. Maintains conversation history across multiple interactions
2. Implements token-based message trimming for context management
3. Uses streaming responses for real-time user experience
4. Implements a graph-based workflow with LangGraph
5. Provides persistent memory using MemorySaver

Key concepts covered:
- MessagesState for conversation management
- Message trimming to stay within token limits
- Streaming responses with visual feedback
- Memory checkpointing for conversation persistence
- Interactive command-line interface

This builds upon the simple LLM application by adding conversational capabilities
and memory management, making it suitable for extended interactions.
"""

# Import necessary libraries
import os  # Operating system interface for environment variables
from time import sleep  # For controlling streaming response speed
from langchain_openai import AzureChatOpenAI  # Azure OpenAI integration
from langchain_core.messages import HumanMessage, trim_messages, AIMessage, SystemMessage  # Message types
from langgraph.checkpoint.memory import MemorySaver  # Persistent conversation memory
from langgraph.graph import START, MessagesState, StateGraph  # Graph-based workflow
from dotenv import load_dotenv  # Environment variable loader

# Load environment variables from .env file
# This loads Azure OpenAI credentials and configuration
load_dotenv()

# ============================================================================
# STEP 1: Initialize Azure OpenAI Model
# ============================================================================

# Initialize the AzureChatOpenAI model with environment variables
# This establishes connection to your Azure OpenAI resource for chat completions
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],        # Your Azure OpenAI endpoint URL
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], # Name of your deployed model
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"], # API version for compatibility
    model_name=os.environ["AZURE_OPENAI_MODEL_NAME"],          # Model name (e.g., gpt-4)
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],         # Your API key for authentication
)

# ============================================================================
# STEP 2: Define Conversation Workflow
# ============================================================================

# Define the state schema for the workflow using MessagesState
# MessagesState automatically manages a list of conversation messages
workflow = StateGraph(state_schema=MessagesState)

# ============================================================================
# STEP 3: Configure Message Trimming for Token Management
# ============================================================================

# Define a trimmer to limit the number of tokens in the messages
# This prevents exceeding model context limits and controls costs
trimmer = trim_messages(
    max_tokens=65,          # Maximum tokens to keep in conversation history
    strategy="last",        # Keep the most recent messages
    token_counter=model,    # Use the model's tokenizer for accurate counting
    include_system=True,    # Always preserve system messages
    allow_partial=False,    # Don't allow cutting messages in the middle
    start_on="human",       # Start trimming from human messages
)

# ============================================================================
# STEP 4: Define Model Interaction Function
# ============================================================================

def call_model(state: MessagesState):
    """
    Process conversation state and generate AI response
    
    This function:
    1. Takes the current conversation state (list of messages)
    2. Trims messages to stay within token limits
    3. Sends trimmed messages to the AI model
    4. Returns the AI response to be added to the conversation
    
    Args:
        state (MessagesState): Current conversation state containing message history
        
    Returns:
        dict: Dictionary with "messages" key containing the AI response
    """
    # Trim messages to stay within token limits while preserving context
    trimmed_messages = trimmer.invoke(state["messages"])
    
    # Generate response from the AI model
    response = model.invoke(trimmed_messages)
    
    # Return the response in the expected format for state updates
    return {"messages": response}

# ============================================================================
# STEP 5: Build Conversation Workflow Graph
# ============================================================================

# Add the start node and the model call to the workflow
# This creates a simple linear flow: START → model → END
workflow.add_edge(START, "model")

# Add the call_model function as a node in the workflow
# This node will handle AI response generation
workflow.add_node("model", call_model)

# ============================================================================
# STEP 6: Setup Memory and Compile Application
# ============================================================================

# Create a memory saver to keep track of the conversation history
# This will allow the workflow to remember previous messages and responses
# MemorySaver stores conversation state in memory (lost when app restarts)
memory = MemorySaver()

# Compile the workflow with memory checkpointing enabled
# This creates a stateful application that can maintain conversation context
app = workflow.compile(checkpointer=memory)

# Configuration for the application
# thread_id groups related conversations together
config = {"configurable": {"thread_id": "abc123"}}

# ============================================================================
# STEP 7: Interactive Chat Loop
# ============================================================================

print("🤖 Chatbot Application Started!")
print("💡 This chatbot has memory and will remember our conversation.")
print("💡 It also streams responses in real-time for better user experience.")
print("💡 The AI has been configured with a special personality (haiku samurai poet).")
print("Type 'exit' to quit.")
print("-" * 60)

# Start the interactive loop to chat with the model
while True:
    # Get user input
    query = input("You: ")
    if query.lower() == "exit":
        print("👋 Goodbye! Chat session ended.")
        break

    # Prepare messages for this interaction
    # SystemMessage defines the AI's personality and behavior
    # HumanMessage contains the user's current input
    input_messages = [
        SystemMessage("You are a poet, who is always responding in haiku samurai style"),
        HumanMessage(query)
    ]

    print("🤖 AI: ", end="", flush=True)

    # Stream the response for real-time user experience
    # stream_mode="messages" gives us individual message chunks
    stream = app.stream({"messages": input_messages}, config, stream_mode="messages")

    # Process each chunk of the streaming response
    for chunk, metadata in stream:
        if isinstance(chunk, AIMessage):  # Filter to just model responses
            sleep(0.2)  # Add slight delay for visual effect
            print(chunk.content, end="", flush=True)  # Print without newline
    print()  # New line after the response
    print("-" * 40)



"""
Key Concepts Demonstrated:

1. CONVERSATION MEMORY:
   - MessagesState automatically tracks conversation history
   - MemorySaver provides persistent memory across interactions
   - Thread IDs group related conversations together

2. TOKEN MANAGEMENT:
   - trim_messages() prevents exceeding model context limits
   - Configurable token limits with different trimming strategies
   - Preserves important messages (system messages) while trimming others

3. STREAMING RESPONSES:
   - Real-time response generation for better user experience
   - Chunk-by-chunk processing of AI responses
   - Visual feedback with controlled timing

4. GRAPH-BASED WORKFLOWS:
   - LangGraph provides structured conversation flow
   - State management with automatic message tracking
   - Checkpointing enables conversation persistence

5. PERSONALITY AND CONTEXT:
   - System messages define AI behavior and personality
   - Consistent character maintenance across interactions
   - Context-aware responses based on conversation history

Benefits of this approach:
- ✅ Maintains conversation context and memory
- ✅ Efficient token usage with smart trimming
- ✅ Engaging user experience with streaming
- ✅ Scalable architecture with graph-based workflows
- ✅ Persistent conversations with checkpointing

Advanced Features:
- Message trimming prevents token limit exceeded errors
- Streaming provides immediate feedback to users
- Memory enables multi-turn conversations
- Graph structure allows for complex conversation flows

Try these interactions:
1. Ask the AI to remember something about you
2. Reference something from earlier in the conversation
3. Ask for haiku poems about different topics
4. Notice how the AI maintains its samurai poet personality
"""




