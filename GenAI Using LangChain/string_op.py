from langchain_huggingface import HuggingFaceEndpoint , HuggingFacePipeline , ChatHuggingFace
from langchain.langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B",
    task="conversational",
    temperature=2.0
)

model=ChatHuggingFace(llm=llm)