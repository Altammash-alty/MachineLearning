from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
import os

repo_id = "meta-llama/Llama-3.1-8B"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,
    max_new_tokens=256,
)


question = "What is the capital of France?"
print(f"Question: {question}")

response = llm.invoke(question)
print(f"Answer: {response}")

