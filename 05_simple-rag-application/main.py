
"""
RAG (Retrieval-Augmented Generation) Application

This example demonstrates how to build a RAG system that:
1. Loads content from a web page (Azure Build 2025 blog post)
2. Splits the content into manageable chunks
3. Creates embeddings and stores them in a vector database
4. Retrieves relevant chunks based on user questions
5. Uses the retrieved context to generate informed answers

RAG combines the power of retrieval (finding relevant information) with 
generation (creating natural language responses) to answer questions about 
specific documents or knowledge bases.
"""

# Import necessary libraries
import bs4  # Beautiful Soup for HTML parsing
import os   # Operating system interface for environment variables
from langchain import hub  # LangChain Hub for accessing shared prompts
from langchain_community.document_loaders import WebBaseLoader  # Web content loader
from langchain_core.documents import Document  # Document structure
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Text chunking
from langgraph.graph import START, StateGraph  # Graph-based workflow
from typing_extensions import List, TypedDict  # Type hints
from langchain_core.vectorstores import InMemoryVectorStore  # Vector storage
from langchain_openai import AzureOpenAIEmbeddings  # Azure OpenAI embeddings
from langchain_openai import AzureChatOpenAI  # Azure OpenAI chat model
from dotenv import load_dotenv  # Environment variable loader

# Load environment variables from .env file
# This loads Azure OpenAI credentials and other configuration
load_dotenv()

# ============================================================================
# STEP 1: Initialize Azure OpenAI Models
# ============================================================================

# Initialize the Language Model (LLM) for generating responses
# This will be used to create the final answer based on retrieved context
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

# Initialize the Embeddings model for converting text to vectors
# This creates numerical representations of text that can be compared for similarity
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

# Create an in-memory vector store to hold our document embeddings
# This will allow us to perform similarity searches to find relevant content
vector_store = InMemoryVectorStore(embeddings)

# ============================================================================
# STEP 2: Load and Process Web Content
# ============================================================================

# Load content from Azure's Build 2025 announcement blog post
# WebBaseLoader fetches web content and converts it to LangChain documents
loader = WebBaseLoader(
    web_paths=("https://azure.microsoft.com/en-us/blog/all-the-azure-news-you-dont-want-to-miss-from-microsoft-build-2025/",),
    bs_kwargs=dict(
        # Use Beautiful Soup to parse only specific content blocks
        # This filters out navigation, ads, and other non-content elements
        parse_only=bs4.SoupStrainer(
            class_=("wp-block-list-item", "wp-block-post-title", "wp-block-post-header")
        )
    ),
)

# Load the documents from the web page
print("Loading web content...")
docs = loader.load()
print(f"Loaded {len(docs)} document(s)")

# ============================================================================
# STEP 3: Split Documents into Chunks
# ============================================================================

# Split the loaded documents into smaller chunks for better retrieval
# Smaller chunks allow for more precise matching of relevant content
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # Maximum characters per chunk
    chunk_overlap=200    # Overlap between chunks to maintain context
)

print("Splitting documents into chunks...")
all_splits = text_splitter.split_documents(docs)
print(f"Created {len(all_splits)} text chunks")

# ============================================================================
# STEP 4: Create Vector Embeddings and Store in Vector Database
# ============================================================================

# Convert text chunks to embeddings and store them in the vector database
# This enables semantic similarity search - finding chunks that are 
# conceptually similar to the user's question
print("Creating embeddings and populating vector store...")
_ = vector_store.add_documents(documents=all_splits)
print("Vector store populated successfully!")

# ============================================================================
# STEP 5: Setup RAG Prompt Template
# ============================================================================

# Load a pre-built prompt template optimized for RAG applications
# This prompt tells the LLM how to use the retrieved context to answer questions
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")


# ============================================================================
# STEP 6: Define RAG Application State and Workflow
# ============================================================================

# Define the state structure for our RAG application
# This tracks the flow of data through our retrieve → generate pipeline
class State(TypedDict):
    question: str           # User's input question
    context: List[Document] # Retrieved relevant documents
    answer: str            # Final generated answer


# ============================================================================
# STEP 7: Define RAG Pipeline Functions
# ============================================================================

def retrieve(state: State):
    """
    RETRIEVAL STEP: Find relevant document chunks based on the user's question
    
    This function:
    1. Takes the user's question from the state
    2. Performs similarity search in the vector store
    3. Returns the most relevant document chunks as context
    """
    print(f"🔍 Retrieving relevant documents for: '{state['question']}'")
    
    # Perform similarity search to find relevant chunks
    # This compares the question's embedding with stored document embeddings
    retrieved_docs = vector_store.similarity_search(state["question"])
    
    print(f"📄 Found {len(retrieved_docs)} relevant document chunks")
    return {"context": retrieved_docs}


def generate(state: State):
    """
    GENERATION STEP: Create an answer using the retrieved context
    
    This function:
    1. Combines all retrieved document chunks into a single context string
    2. Uses the RAG prompt template with the question and context
    3. Generates a natural language answer using the LLM
    """
    print("🤖 Generating answer using retrieved context...")
    
    # Combine all retrieved document chunks into a single context string
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Create the prompt with question and context using the RAG template
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    
    # Generate the response using the LLM
    response = llm.invoke(messages)
    
    print("✅ Answer generated successfully!")
    return {"answer": response.content}


# ============================================================================
# STEP 8: Build and Execute RAG Workflow
# ============================================================================

# Create a graph-based workflow that connects retrieval and generation steps
# LangGraph allows us to define complex workflows as directed graphs
print("🔧 Building RAG workflow...")

# Build the workflow graph with our retrieve → generate sequence
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")  # Start with retrieval step
graph = graph_builder.compile()

print("🔧 RAG workflow compiled successfully!")

# ============================================================================
# STEP 9: Test the RAG System
# ============================================================================

print("\n" + "="*60)
print("🚀 TESTING RAG SYSTEM")
print("="*60)

# Test the RAG system with a question about Build 2025 announcements
test_question = "what has been announced at Build 2025?"
print(f"❓ Question: {test_question}")
print("\n" + "-"*60)

# Execute the RAG workflow
response = graph.invoke({"question": test_question})

# Display the final answer
print("🎯 ANSWER:")
print(response["answer"])
print("\n" + "="*60)

# ============================================================================
# How to Use This RAG System:
# ============================================================================
"""
This RAG system works in the following steps:

1. DOCUMENT LOADING: Fetches content from a web page using WebBaseLoader
2. TEXT SPLITTING: Breaks the content into manageable chunks for better retrieval
3. EMBEDDING CREATION: Converts text chunks to vector embeddings using Azure OpenAI
4. VECTOR STORAGE: Stores embeddings in an in-memory vector database
5. QUESTION PROCESSING: When a question is asked:
   a. RETRIEVAL: Finds the most relevant chunks using similarity search
   b. GENERATION: Uses retrieved context to generate an informed answer

Key Benefits of RAG:
- ✅ Provides answers based on specific, up-to-date information
- ✅ Reduces hallucination by grounding responses in real content
- ✅ Can work with any text-based knowledge source
- ✅ Combines the power of search with natural language generation

To modify this system:
- Change the web_paths in WebBaseLoader to use different sources
- Adjust chunk_size and chunk_overlap for different document types
- Modify the CSS selectors in bs_kwargs for different website structures
- Replace InMemoryVectorStore with persistent storage for production use
"""