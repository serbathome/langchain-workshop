# Import relevant functionality
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create the agent
# This example uses the Tavily Search tool to search for information
# and the Anthropic Claude model to generate responses.
# The agent will remember the conversation history using MemorySaver.

# Create a memory saver to keep track of the conversation history
memory = MemorySaver()

# Initialize the model
model = ChatAnthropic(model_name="claude-3-7-sonnet-latest")

# Create the search tool
# Tavily Search is a tool that allows the agent to search the web for information.
search = TavilySearchResults(max_results=2)

# Create the agent executor with the model, tools, and memory saver
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}

# The agent will respond to user messages and remember the conversation history.
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# The agent can also answer questions based on the conversation history.
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

