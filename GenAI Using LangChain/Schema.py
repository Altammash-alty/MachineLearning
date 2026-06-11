from langchain_core import PydanticOutpuParser
from Pydantic import BaseModel , Field
from Typing import Literal


class Person(BaseModel):
    age : int = Field("description":"", examples="")
    name : str = Field("description":"",examples="")
    review : str = Field("description":"",examples="",Literal:["pos","neg","neu"])