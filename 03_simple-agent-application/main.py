
"""
ReAct Agent Application with Web Search

This example demonstrates how to build an intelligent agent that can:
1. Reason about user queries and decide what actions to take
2. Use external tools (web search) to gather information
3. Act on the information to provide comprehensive answers
4. Maintain conversation memory across interactions
5. Demonstrate the ReAct (Reasoning and Acting) pattern

Key concepts covered:
- ReAct agent pattern (Reasoning → Acting → Observing)
- Tool integration with Tavily web search
- Anthropic Claude model for advanced reasoning
- Autonomous decision-making and tool selection
- Multi-step problem solving with external data

This represents a significant advancement from simple LLM interactions to
autonomous AI agents that can interact with the real world through tools.
"""

# Import relevant functionality
from langchain_anthropic import ChatAnthropic  # Anthropic Claude model for reasoning
from langchain_community.tools.tavily_search import TavilySearchResults  # Web search tool
from langchain_core.messages import HumanMessage  # Message types
from langgraph.checkpoint.memory import MemorySaver  # Persistent conversation memory
from langgraph.prebuilt import create_react_agent  # Pre-built ReAct agent implementation
from dotenv import load_dotenv  # Environment variable loader

# Load environment variables from .env file
# This loads API keys for Anthropic and Tavily services
load_dotenv()

# ============================================================================
# STEP 1: Understanding the ReAct Agent Pattern
# ============================================================================
"""
ReAct (Reasoning and Acting) is a paradigm where AI agents:
1. REASONING: Think through the problem and plan actions
2. ACTING: Execute actions using available tools
3. OBSERVING: Analyze results and decide on next steps

This creates autonomous agents that can solve complex, multi-step problems
by combining their reasoning capabilities with real-world tool usage.
"""

# ============================================================================
# STEP 2: Create Agent Components
# ============================================================================

# Create a memory saver to keep track of the conversation history
# This allows the agent to remember previous interactions and build context
memory = MemorySaver()

# Initialize the Anthropic Claude model for agent reasoning
# Claude excels at reasoning tasks and following complex instructions
model = ChatAnthropic(model_name="claude-3-7-sonnet-latest")

# Create the Tavily search tool for web information retrieval
# Tavily provides high-quality, AI-optimized search results
search = TavilySearchResults(
    max_results=2  # Limit results to keep responses focused and manageable
)

print("🔧 Initializing ReAct Agent...")
print(f"🧠 Model: {model.model_name}")
print(f"🔍 Search Tool: Tavily (max {search.max_results} results)")
print("💾 Memory: Enabled with MemorySaver")
print("-" * 60)

# ============================================================================
# STEP 3: Create and Configure the ReAct Agent
# ============================================================================

# Create the agent executor with the model, tools, and memory saver
# The create_react_agent function builds a complete agent workflow that can:
# - Analyze user input and decide what actions to take
# - Use tools when additional information is needed
# - Combine tool results with its knowledge to provide comprehensive answers
tools = [search]  # List of available tools for the agent
agent_executor = create_react_agent(
    model,              # The reasoning model (Claude)
    tools,              # Available tools (Tavily search)
    checkpointer=memory # Memory for conversation persistence
)

# Configuration for agent conversations
# thread_id groups related conversations and maintains context
config = {"configurable": {"thread_id": "abc123"}}

print("✅ ReAct Agent created successfully!")
print("🎯 Agent capabilities:")
print("   - Reasoning about complex queries")
print("   - Web search for real-time information") 
print("   - Memory of conversation context")
print("   - Multi-step problem solving")
print("=" * 60)

# ============================================================================
# STEP 4: Demonstrate Agent Reasoning and Memory
# ============================================================================

print("🚀 DEMONSTRATION 1: Building Context")
print("The agent will remember information about the user...")

# First interaction: Establish context
# The agent will remember the conversation history using MemorySaver
print("\n👤 User: hi im bob! and i live in sf")

for step in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]},
    config,
    stream_mode="values",
):
    # Pretty print the last message (agent's response)
    step["messages"][-1].pretty_print()

print("-" * 60)

# ============================================================================
# STEP 5: Demonstrate Agent Tool Usage and Reasoning
# ============================================================================

print("🚀 DEMONSTRATION 2: ReAct Pattern in Action")
print("Watch how the agent:")
print("1. 🤔 REASONS: Analyzes the query and remembers Bob lives in SF")
print("2. 🔍 ACTS: Uses web search to find current weather information") 
print("3. 👀 OBSERVES: Processes search results and provides a complete answer")

# Second interaction: Query that requires tool usage and memory
# The agent can also answer questions based on the conversation history
print("\n👤 User: whats the weather where I live?")

for step in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]},
    config,
    stream_mode="values",
):
    # Pretty print the last message to see the agent's reasoning process
    step["messages"][-1].pretty_print()

print("=" * 60)

# ============================================================================
# Key Concepts and Learning Outcomes:
# ============================================================================
"""
🎓 What You've Learned:

1. REACT PATTERN:
   - Reasoning: Agent analyzes queries and plans actions
   - Acting: Agent executes tools when additional information is needed
   - Observing: Agent processes tool results and synthesizes responses

2. AUTONOMOUS DECISION MAKING:
   - Agent decides when to use tools vs. existing knowledge
   - Multi-step reasoning to solve complex problems
   - Context-aware responses based on conversation history

3. TOOL INTEGRATION:
   - Tavily search provides real-time web information
   - Seamless integration between reasoning and external data
   - Tool results are intelligently incorporated into responses

4. CONVERSATION MEMORY:
   - Agent remembers user information (Bob lives in SF)
   - Context carries forward across multiple interactions
   - Persistent memory enables natural, flowing conversations

5. ADVANCED AI CAPABILITIES:
   - Beyond simple question-answering to intelligent assistance
   - Real-world problem solving with external information sources
   - Foundation for building sophisticated AI applications

🔧 Architecture Benefits:
- ✅ Autonomous reasoning and decision-making
- ✅ Real-time information access through web search
- ✅ Persistent conversation memory and context
- ✅ Extensible tool framework for additional capabilities
- ✅ Natural language interaction with complex workflows

🚀 Next Level Applications:
- Add more tools (calculators, databases, APIs)
- Implement custom tools for specific domains
- Build multi-agent systems with specialized roles
- Create production agents with error handling and monitoring

Try asking the agent:
- Questions that require recent information
- Queries that reference previous conversation context
- Complex, multi-step problems that need reasoning
- Anything that combines memory with real-time data!
"""

