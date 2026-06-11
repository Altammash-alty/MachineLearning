from langchain_huggingface import HuggingFacePipeline , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel , RunnableBranch , RunnableLambda

from dotenv import load_dotenv