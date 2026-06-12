import random


class NakliLLM():
    def __init__(self):
        print("LLM Created")
    def predict(self,prompt):

        response_list = [
            """
            Name : Altammash
            Age : 21
            Review : "I am a good boy."
            """,

            """
            Name : Zoya
            Age : 19
            Review : "I am a good girl."
            """,

            """
            Name : Aiman
            Age : 25
            Review : "I am a good girl."
            """,
        ]

        return { "response" : random.choice(response_list)}


class NakliPromptTemplate():
    def __init__(self,template,input_variables):
        self.template=template
        self.input_variables=input_variables

    def format(self,dict):
       return self.template.format(**dict)


template="""
Create a small description of this person
Name : {name}
Age : {age}
Review : {review}
"""