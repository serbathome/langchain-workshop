# LangChain Workshop

A hands-on workshop for learning LangChain fundamentals with Azure OpenAI, building AI applications, and implementing agent patterns with Large Language Models (LLMs).

## 📋 Content Overview

This workshop includes five progressive examples:

### 01_simple-LLM-application
- **Azure OpenAI Integration**: Direct integration with Azure OpenAI using environment variables
- **Basic LLM Interactions**: Simple message handling and prompt templates
- **Translation Example**: English to Italian and German translation using system prompts

### 02_chatbot-application  
- **Interactive Chatbot**: Command-line chatbot with conversation loop
- **Memory Management**: Persistent conversation history using MemorySaver
- **Message Trimming**: Token management with configurable message limits
- **Streaming Responses**: Real-time response streaming with visual feedback

### 03_simple-agent-application
- **ReAct Agent**: Reasoning and Acting agent pattern implementation  
- **Web Search Integration**: Tavily search tool for real-time information
- **Conversation Memory**: Thread-based conversation tracking
- **Tool Usage**: Autonomous tool selection and execution

### 04_native-function-agent-application
- **Function Calling**: Native Python functions as LLM-callable tools
- **Custom Tool Creation**: Convert Python functions to agent tools automatically
- **Multi-Step Operations**: Sequential function execution with intermediate results
- **Graph Workflows**: Conditional routing between assistant and tool execution

### 05_simple-rag-application
- **RAG Implementation**: Retrieval-Augmented Generation with web content
- **Web Content Loading**: Extract and process content from Azure blogs
- **Vector Embeddings**: Create and store document embeddings for similarity search
- **Document Chunking**: Split large documents into manageable pieces
- **Context-Aware Answers**: Generate responses based on retrieved relevant content

## 🔧 Requirements

### System Requirements
- Python 3.8 or higher
- Git
- Code editor (VS Code recommended)

### Python Dependencies
Install required packages using pip:
```bash
pip install -r requirements.txt
```

The workshop uses the following key dependencies:
- **langchain**: Core LangChain framework
- **langchain-openai**: Azure OpenAI integration
- **langchain-anthropic**: Anthropic Claude integration (for agents)
- **langchain-core**: Core LangChain components
- **langgraph**: Graph-based agent framework
- **langchain-community**: Community tools including Tavily search and web loaders
- **tavily-python**: Web search API integration
- **langgraph-checkpoint-sqlite**: Persistent memory storage
- **python-dotenv**: Environment variable management
- **beautifulsoup4**: HTML parsing for web content extraction
- **langchain-text-splitters**: Document chunking utilities

## 🛠️ Environment Setup

### Virtual Environment (Recommended)

**Creating and activating a virtual environment:**

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment  
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Deactivating virtual environment:**
```bash
deactivate
```

### Environment Variables (.env)

Create a `.env` file in the project root to store your API keys and configuration:

```env
# Azure OpenAI Configuration (Required)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_MODEL_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Azure OpenAI Embeddings (Required for RAG Example)
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=your_embeddings_deployment_name

# LangSmith Tracing (Optional)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=langchain-workshop

# Anthropic API (Required for Agent Example)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Tavily Search API (Required for Agent Example)
TAVILY_API_KEY=your_tavily_api_key_here
```

**⚠️ Important Security Notes:**
- **NEVER** commit your `.env` file to version control
- Keep your API keys local and secure
- The `.env` file should be added to `.gitignore`
- Consider using environment-specific configurations for different stages

## 🚀 Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/langchain-workshop.git
   cd langchain-workshop
   ```

2. **Set up virtual environment and install dependencies:**
   ```powershell
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   ```bash
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   # Create .env file with your API keys (see Environment Variables section)
   # Ensure you have Azure OpenAI, Anthropic, and Tavily API keys
   ```

4. **Run workshop examples:**
   ```bash
   # Example 1: Simple LLM with Azure OpenAI
   python 01_simple-LLM-application/main.py
   
   # Example 2: Interactive chatbot with memory
   python 02_chatbot-application/main.py
   
   # Example 3: ReAct agent with web search
   python 03_simple-agent-application/main.py
   
   # Example 4: Function-calling agent with custom tools
   python 04_native-function-agent-application/main.py
   
   # Example 5: RAG system with web content
   python 05_simple-rag-application/main.py
   ```

## 🔍 LangSmith Tracing

LangSmith provides powerful debugging and monitoring capabilities for LangChain applications.

### Setup LangSmith
1. Sign up at [LangSmith](https://smith.langchain.com)
2. Get your API key from the settings
3. Add to your `.env` file:
   ```env
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=your_langsmith_api_key
   LANGSMITH_PROJECT=langchain-workshop
   ```

### Required API Keys

To run all workshop examples, you'll need:

1. **Azure OpenAI** (Required for examples 1, 2, 4 & 5):
   - Create an Azure OpenAI resource in Azure Portal
   - Deploy a GPT-4 model (for chat completions)
   - Deploy an embeddings model (text-embedding-ada-002 for RAG example)
   - Get endpoint, API key, and deployment names

2. **Anthropic Claude** (Required for example 3):
   - Sign up at [Anthropic](https://console.anthropic.com)
   - Get your API key from the console

3. **Tavily Search** (Required for example 3):
   - Sign up at [Tavily](https://tavily.com)
   - Get your API key for web search functionality

## 📚 Workshop Examples

### Example 1: Simple LLM Application (`01_simple-LLM-application/`)
**Focus**: Basic Azure OpenAI integration and prompt templates

**Features**:
- Direct Azure OpenAI model invocation
- System and user message handling
- Prompt templates with variable substitution
- Translation examples (English → Italian, German)

**Run**: `python 01_simple-LLM-application/main.py`

### Example 2: Chatbot Application (`02_chatbot-application/`)
**Focus**: Interactive conversational AI with memory

**Features**:
- Command-line chat interface
- Persistent conversation memory using MemorySaver
- Token-based message trimming for context management
- Real-time streaming responses with visual feedback
- Conversation state management

**Run**: `python 02_chatbot-application/main.py`
**Usage**: Type messages and get responses. Type 'exit' to quit.

### Example 3: Simple Agent Application (`03_simple-agent-application/`)
**Focus**: ReAct (Reasoning and Acting) agent with tool usage

**Features**:
- Anthropic Claude model for agent reasoning
- Tavily web search tool integration
- Autonomous tool selection and execution
- Memory-persistent agent conversations
- Multi-step reasoning and web search capabilities

**Run**: `python 03_simple-agent-application/main.py`
**Example queries**: 
- "Hi, I'm Bob and I live in SF"
- "What's the weather where I live?"

### Example 4: Native Function Agent Application (`04_native-function-agent-application/`)
**Focus**: Function calling with custom Python tools

**Features**:
- Native Python functions converted to LLM-callable tools
- Custom tool creation with automatic schema generation
- Multi-step arithmetic operations with function chaining
- Graph-based workflow with conditional tool routing
- Sequential function execution with intermediate results

**Run**: `python 04_native-function-agent-application/main.py`
**Example output**: Performs multi-step calculations: "Add 3 and 4. Multiply the output by 2. Divide the output by 5"

### Example 5: Simple RAG Application (`05_simple-rag-application/`)
**Focus**: Retrieval-Augmented Generation with web content

**Features**:
- Web content loading from Azure Build 2025 blog posts
- Document chunking with RecursiveCharacterTextSplitter
- Vector embeddings using Azure OpenAI Embeddings
- In-memory vector store for similarity search
- Context-aware answer generation using retrieved documents
- Graph-based RAG workflow with LangGraph

**Run**: `python 05_simple-rag-application/main.py`
**Example output**: Answers questions about Azure Build 2025 announcements based on web content

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

**2. API Key Issues**
- Verify your `.env` file is in the project root
- Check API key format and validity (no extra spaces or quotes)
- Ensure Azure OpenAI deployment name matches your actual deployment
- Verify Azure OpenAI endpoint URL format

**3. Azure OpenAI Connection Issues**
```bash
# Verify your Azure OpenAI configuration
# Check deployment name, endpoint, and API version
# Ensure your resource has the correct model deployed
```

**4. Agent/Search Tool Issues**
- Verify Tavily API key is valid and has remaining credits
- Check Anthropic API key if agent fails to respond
- Ensure internet connection for web search functionality

**5. Virtual Environment Issues**
```powershell
# Windows PowerShell - if activation fails
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

```bash
# If virtual environment is corrupted, recreate it
rm -rf .venv  # or Remove-Item .venv -Recurse -Force on Windows
python -m venv .venv
# Then activate and reinstall dependencies
```

**6. Import or Module Errors**
```bash
# Ensure virtual environment is activated
# Reinstall dependencies if needed
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```
# If activation fails, try:
python -m venv --clear langchain-env
# Then recreate and activate
```

## 📚 Useful Resources

### Documentation
- [LangChain Documentation](https://python.langchain.com/)
- [LangChain API Reference](https://api.python.langchain.com/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)


### API Providers
- [OpenAI API](https://platform.openai.com/)
- [Anthropic API](https://console.anthropic.com/)


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


