from langchain_huggingface import HuggingFacePipeline , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda
from langchain_core import Pydantic
from Pydantic import BaseModel , Field
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="",
    task="",
    temperature=2
)