from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate



load_dotenv()


template = PromptTemplate(
    template="",
    input_variables="",
    validate_template=True
)

repo_id = "meta-llama/Llama-3.1-8B"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,
    max_new_tokens=256,
)


prompt = template.invoke(
    userinput1="",
    uerinput2="",
    userinput3=""

)

result=llm.invoke(prompt)
print(result.content)

