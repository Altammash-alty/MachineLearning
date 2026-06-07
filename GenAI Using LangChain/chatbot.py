from langchain_huggingface import HuggingFaceEndpoint , HuggingFacePipeline
from langchain.langchain_core.prompts import PromptTemplate

form dotenv import load_dotenv
load_dotenv()

model = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B",
    temperature=0.7,
    max_new_tokens=256,
)
while True :
    user_input=input('Your input'),
    if user_input=='exit' :
        break
    result = model.invoke(user_input)
    print(result.content)