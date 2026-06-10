from langchain_huggingface import HuggingFaceEndpoint , HuggingFacePipeline , ChatHuggingFace
from langchain.langchain_core.prompts import PromptTemplate , HumanMessage , AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B",
    task="conversational",
    temperature=2.0
)

model=ChatHuggingFace(llm=llm)


#1st prompt 

Prompt1=PromptTemplate(
    template="Write a detailed description on the {topic}",
    input_variables=["topic"],
    validate_template=True
)
#2nd prompt
Prompt2=PromptTemplate(
    template="Write a 5 line summary on the following {text}",
    input_variables=["text"],
    validate_template=True
)

parser = StrOutputParser()

#chain creation 

chain = Prompt1 | model | parser | Prompt2 | model | parser

# result=Prompt1.invoke({"topic":"Black Hole"})

# final_result=Prompt2.invoke({"text":result.content})

# print(final_result.content)

