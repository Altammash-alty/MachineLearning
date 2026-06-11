from langchain_huggingface import HuggingFacePipeline , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda
from langchain_core import Pydantic
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


Prompt1 = PromptTemplate(
    template="You are an expert in the {domain} . YOu have to explain the {topic} to the user in a manner such that a small kid of 2 yrs can understand the text language if the explanation is read to him" /n {format_instructions}
    input_variables={"domain","topic"},
)
parser = PydanticOutpuParser(Pydantic_object=Person)

chain = Prompt1 | model | parser
