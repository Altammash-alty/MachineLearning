from langchain_huggingface import HuggingFacePipeline , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda
from langchain_core import Pydantic
from Pydantic import BaseModel , Field
from dotenv import load_dotenv
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="",
    task="",
    temperature=2
)

class Person():
    age : int = Field("description":"", examples="")
    name : str = Field("description":"",examples="")
    review : str = Field("description":"",examples="",Literal:["pos","neg","neu"])
parser = PydanticOutpuParser(Pydantic_object=Person)

chain = Prompt1 | model | parser
