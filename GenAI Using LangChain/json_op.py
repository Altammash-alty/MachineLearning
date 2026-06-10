from langchain_huggingface import HuggingFaceEndpoint , HuggingFacePipeline , ChatHuggingFace
from langchain.langchain_core.prompts import PromptTemplate , HumanMessage , AIMessage
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B",
    task="conversational",
    temperature=2.0
)

model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()
#1st prompt 

Prompt1=PromptTemplate(
    template="Give me th e name , age and country of a fictional character \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':'parser.get_format_instructions()'},
    validate_template=True
)