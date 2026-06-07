from langchain_huggingface import HuggingFaceEndpoint , HuggingFacePipeline
from langchain.langchain_core.prompts import PromptTemplate
from langchain.core_messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B",
    temperature=0.7,
    max_new_tokens=256,
)
messages=[
    SystemMessage(content=""),
    HumanMessag(content="")
]

chat_history = []
while True :
    user_input=input('Your input'),
    messages.append(HumanMessage(content=user_input))
    if user_input=='exit' :
        break
    result = model.invoke(messages)    
    messages.append(AIMessage(content=result.content))
    print(result.content)