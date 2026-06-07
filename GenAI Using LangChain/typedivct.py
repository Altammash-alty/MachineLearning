from typing import TypedDict , Annotated , Optional , Literal

class Person(TypedDict):
    name:str
    age:int

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

#Schema
class Review(TypedDict):
    revie:Annotated[Optional[str],'Explain the data']
    sentiment:Annotated[Literal["pos","neg"]]
    summary:str

structured_model=model.with_structured_output(Person)
result=structured_model.invoke(Prompt)

print(result.content)