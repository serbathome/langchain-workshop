import os
from time import sleep
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, trim_messages, AIMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the AzureChatOpenAI model with environment variables
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    model_name=os.environ["AZURE_OPENAI_MODEL_NAME"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

# Define the state schema for the workflow
workflow = StateGraph(state_schema=MessagesState)

# Define a trimmer to limit the number of tokens in the messages
# This will keep the last 65 tokens, including system messages, and will not allow partial messages.
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Define a function to call the model with the trimmed messages
def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    response = model.invoke(trimmed_messages)
    return {"messages": response}

# Add the start node and the model call to the workflow
workflow.add_edge(START, "model")
# Add the call_model function as a node in the workflow
workflow.add_node("model", call_model)

# Create a memory saver to keep track of the conversation history
# This will allow the workflow to remember previous messages and responses.
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Configuration for the application
# This can include any configurable parameters needed for the workflow.
config = {"configurable": {"thread_id": "abc123"}}


print("Type 'exit' to quit.")

# Start the interactive loop to chat with the model
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    input_messages = [HumanMessage(query)]

    stream = app.stream({"messages": input_messages},config,stream_mode="messages")

    for chunk, metadata in stream:
        if isinstance(chunk, AIMessage):  # Filter to just model responses
            sleep(0.2)
            print(chunk.content, end="", flush=True)
    print()  # New line after the response







