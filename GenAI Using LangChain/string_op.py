from langchain_huggingface import HuggingFaceEndpoint , HuggingFacePipeline
from langchain.langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B",
    task="",
    temperature=2.0
)