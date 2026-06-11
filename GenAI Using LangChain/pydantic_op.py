class Person(BaseModel):
    name : str
    age : int = Field()
    city : str = FIeld()