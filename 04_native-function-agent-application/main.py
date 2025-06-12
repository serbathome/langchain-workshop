
"""
Native Function Agent Application with Custom Tools

This example demonstrates how to build an agent that uses native Python functions as tools:
1. Define custom Python functions as agent tools
2. Bind tools directly to the LLM for function calling
3. Create a graph-based workflow for tool execution
4. Handle multi-step arithmetic operations with tool chaining
5. Implement conditional routing between assistant and tools

Key concepts covered:
- Function calling with Azure OpenAI
- Custom tool creation from Python functions
- Tool binding and parallel execution control
- Graph-based agent workflows with conditional edges
- Multi-step problem solving with function chaining

This approach differs from external tools (like web search) by using local Python functions,
making it ideal for computational tasks, data processing, and domain-specific operations.
"""

# Import necessary libraries
import os  # Operating system interface for environment variables
from dotenv import load_dotenv  # Environment variable loader
from langchain_openai import AzureChatOpenAI  # Azure OpenAI integration
from langgraph.graph import MessagesState, START, StateGraph  # Graph-based workflow
from langchain_core.messages import HumanMessage, SystemMessage  # Message types
from langgraph.prebuilt import tools_condition, ToolNode  # Pre-built workflow components


# Load environment variables from .env file
# This loads Azure OpenAI credentials and configuration
load_dotenv()

# ============================================================================
# STEP 1: Define Custom Tool Functions
# ============================================================================

def multiply(a: int, b: int) -> int:
    """
    Multiply two integers together.
    
    This function will be converted into an agent tool that the LLM can call
    when it needs to perform multiplication operations.

    Args:
        a: First integer to multiply
        b: Second integer to multiply
    """
    return a * b

def add(a: int, b: int) -> int:
    """
    Add two integers together.
    
    This function will be converted into an agent tool that the LLM can call
    when it needs to perform addition operations.

    Args:
        a: First integer to add
        b: Second integer to add
    """
    return a + b

def divide(a: int, b: int) -> float:
    """
    Divide the first integer by the second integer.
    
    This function will be converted into an agent tool that the LLM can call
    when it needs to perform division operations.

    Args:
        a: Dividend (number to be divided)
        b: Divisor (number to divide by)
    """
    return a / b

# Create a list of tool functions that the agent can use
# These Python functions will be automatically converted to LLM-callable tools
tools = [add, multiply, divide]

print("🔧 Defined Custom Tools:")
for tool in tools:
    print(f"   📊 {tool.__name__}: {tool.__doc__.split('.')[0].strip()}")
print("-" * 60)

# ============================================================================
# STEP 2: Initialize Azure OpenAI Model with Function Calling
# ============================================================================

# Initialize Azure OpenAI model for function calling capabilities
llm = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),        # Your Azure OpenAI endpoint URL
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"), # Name of your deployed model
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"), # API version for compatibility
    openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),         # Your API key for authentication
)

# Bind tools to the LLM to enable function calling
# parallel_tool_calls=False ensures tools are called sequentially, not simultaneously
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

print("🤖 Azure OpenAI Model Configuration:")
print("   ✅ Model: Connected to Azure OpenAI")
print("   🔗 Function Calling: Enabled")
print(f"   🛠️  Bound Tools: {len(tools)} arithmetic functions")
print("   🔄 Parallel Calls: Disabled (sequential execution)")
print("-" * 60)


# ============================================================================
# STEP 3: Define Agent Workflow Components
# ============================================================================

# System message to define the agent's role and behavior
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
    """
    Assistant node that processes user input and decides on tool usage.
    
    This function:
    1. Takes the current conversation state
    2. Combines system message with conversation history
    3. Invokes the LLM with tool-calling capabilities
    4. Returns either a response or tool calls based on the query
    
    Args:
        state (MessagesState): Current conversation state containing message history
        
    Returns:
        dict: Dictionary with "messages" key containing the LLM response
    """
    # Combine system message with conversation history
    messages = [sys_msg] + state["messages"]
    
    # Invoke LLM with tool-calling capabilities
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

print("🧠 Agent Workflow Components:")
print("   🤖 Assistant Node: LLM with function calling")
print("   🛠️  Tools Node: Executes arithmetic functions")
print("   🔀 Conditional Routing: Based on tool call requirements")
print("-" * 60)

# ============================================================================
# STEP 4: Build Function-Calling Agent Graph
# ============================================================================

# Create the state graph for our function-calling agent
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)    # LLM node for reasoning and tool selection
builder.add_node("tools", ToolNode(tools))  # Tool execution node for running functions

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")        # Always start with the assistant

# Add conditional edges for intelligent routing
builder.add_conditional_edges(
    "assistant",
    # tools_condition examines the assistant's response:
    # - If it contains tool calls → route to "tools" node
    # - If it's a final answer → route to END
    tools_condition,
)

# After tools execute, always return to assistant for next step or final response
builder.add_edge("tools", "assistant")

# Compile the graph into an executable workflow
react_graph = builder.compile()

print("🔧 Function-Calling Agent Graph Built:")
print("   📊 Workflow: START → assistant ⇄ tools → assistant → END")
print("   🤔 Logic: Assistant decides when to call tools vs. provide final answer")
print("   🔄 Flow: Supports multi-step operations with tool chaining")
print("=" * 60)

# ============================================================================
# STEP 5: Test Multi-Step Arithmetic Operations
# ============================================================================

print("🚀 TESTING FUNCTION-CALLING AGENT")
print("Demonstrating multi-step arithmetic with function chaining...")

# Create a complex arithmetic query that requires multiple tool calls
test_query = "Add 3 and 4. Multiply the output by 2. Divide the output by 5"
print(f"📝 Query: {test_query}")
print("\n🔄 Expected workflow:")
print("   1. Assistant analyzes the multi-step problem")
print("   2. Calls add(3, 4) → returns 7")
print("   3. Calls multiply(7, 2) → returns 14") 
print("   4. Calls divide(14, 5) → returns 2.8")
print("   5. Provides final answer with explanation")
print("\n" + "─" * 60)

# Execute the agent workflow
messages = [HumanMessage(content=test_query)]
result = react_graph.invoke({"messages": messages})

print("💬 CONVERSATION FLOW:")
print("=" * 60)

# Display all messages in the conversation to see the complete workflow
for i, message in enumerate(result['messages'], 1):
    print(f"\n📩 Message {i}:")
    message.pretty_print()
    print("─" * 40)

print("\n" + "=" * 60)

# ============================================================================
# Key Concepts and Learning Outcomes:
# ============================================================================
"""
🎓 What You've Learned:

1. FUNCTION CALLING:
   - Convert Python functions directly into LLM-callable tools
   - Automatic schema generation from function signatures and docstrings
   - Type hints provide parameter validation and documentation

2. TOOL BINDING:
   - bind_tools() connects functions to the LLM
   - parallel_tool_calls controls execution behavior
   - Tools become available for the LLM to call autonomously

3. GRAPH-BASED WORKFLOWS:
   - StateGraph manages conversation flow and tool execution
   - Conditional edges enable intelligent routing based on responses
   - Multi-step problem solving with automatic tool chaining

4. MULTI-STEP REASONING:
   - Agent breaks down complex problems into smaller steps
   - Sequential tool execution with intermediate results
   - Context preservation across multiple function calls

5. CUSTOM TOOL DEVELOPMENT:
   - Domain-specific functions for specialized tasks
   - Proper documentation enables better LLM understanding
   - Local execution without external API dependencies

🔧 Architecture Benefits:
- ✅ Local function execution (no external API calls)
- ✅ Type-safe operations with Python type hints
- ✅ Automatic tool discovery from function metadata
- ✅ Sequential processing for dependent operations
- ✅ Clear separation between reasoning and computation

🚀 Advanced Applications:
- Data processing and analysis functions
- Mathematical and scientific computations
- Database operations and queries
- File system operations
- Custom business logic implementation

💡 Key Differences from External Tools:
- Native Python functions vs. external APIs
- Immediate execution vs. network calls
- Type safety vs. string-based interfaces
- Local control vs. external dependencies

Try modifying this example:
- Add more arithmetic functions (power, square root, etc.)
- Create data processing tools (sorting, filtering)
- Implement validation functions
- Build domain-specific computational tools
"""