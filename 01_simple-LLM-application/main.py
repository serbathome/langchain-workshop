
# import necessary libraries
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# create an instance of AzureChatOpenAI with environment variables
model = AzureChatOpenAI(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
)

# Define the messages to be sent to the model
messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

# Invoke the model with the messages and print the response
result = model.invoke(messages)
print(result.content)

# Alternatively, using a prompt template
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "German", "text": "hi!"})

# Invoke the model with the prompt and print the response
response = model.invoke(prompt)
print(response.content)

