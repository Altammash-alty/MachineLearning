from langchain_huggingface import HuggingFacePipeline , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda
from langchain_core import PydanticOutpuParser , ResponseSchema
from Pydantic import BaseModel , Field
from dotenv import load_dotenv
from typing import Literal
from Schema import Person

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B",
    task="text-generation",
    temperature=2
)

parser = PydanticOutpuParser(Pydantic_object=Person)


Prompt1 = PromptTemplate(
    template="You are an expert in the {domain} . YOu have to explain the {topic} to the user in a manner such that a small kid of 2 yrs can understand the text language if the explanation is read to him\n{format_instructions}",
    input_variables={"domain","topic"},
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

Prompt2 = PromptTemplate(
    template="You are an excellent {domain} . You are given an explanation of the {topic}. Provide a {n} line summary for the explanation having only the key words from the explanation, no extra words \n {format_instructions}",
    input_variables={"domain","topic","n"},
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

schema = [
    ResponseSchema(name="name",description="The name of the person."),
    ResponseSchema(name="age",description="The age of the person."),
    ResponseSchema(name="review",description="The review of the person.")
]

chain1 = Prompt1 | model | parser
chain2 = Prompt2 | model | parser 


seq_chain = RunnableParallel(
    "chain1" : chain1 ,
    "Chain2" : chain2
)

branch_chain = RunnableBranch(
    ("chain1",Prompt1 | model | parser ),
    ("chain2",Prompt2 | model | parser ),
    (RunnableLambda(lambda x: "No matching")
)

data = seq_chain.invoke({
    "domain":"Quantum Mechanics",
    "topic":"Superposition and Entanglement",
    "n":3
})

print(data)

print(seq_chain.get_graph().print_ascii())