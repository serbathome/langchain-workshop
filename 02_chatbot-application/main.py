import os
from time import sleep
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, trim_messages, AIMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    model_name=os.environ["AZURE_OPENAI_MODEL_NAME"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

workflow = StateGraph(state_schema=MessagesState)

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)


def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    response = model.invoke(trimmed_messages)
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}


print("Type 'exit' to quit.")

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







