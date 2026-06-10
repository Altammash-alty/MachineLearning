from langchain_huggingface import HuggingFaceEndpoint , HuggingFacePipeline , ChatHuggingFace
from langchain.langchain_core.prompts import PromptTemplate , HumanMessage , AIMessage
from langchain.output_parsers import StructureOutputParser , ResponseSchema
from dotenv import load_dotenv
import os

load_dotenv()

model=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B",
    task="conversational",
    temperature=2.0,
)
parser=StructureOutputParser()

schema=[
    ResponseSchema(),ResponseSchema(),ResponseSchema
]

