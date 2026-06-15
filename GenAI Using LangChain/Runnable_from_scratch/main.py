import random
from abc import ABC, abstractmethod

class Runnable(ABC):

    @abstractmethod
    def invoke(self,dict):
        pass


class NakliLLM(Runnable):
    def __init__(self):
        print("LLM Created")


    def invoke(self,prompt):
        
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


class NakliPromptTemplate(Runnable):
    def __init__(self,template,input_variables):
        self.template=template
        self.input_variables=input_variables
        
    def invoke(self,dict):
        return self.template.format(**dict)

    def format(self,dict):
       return self.template.format(**dict)



template=NakliPromptTemplate("""
Create a small description of this person
Name : {name}
Age : {age}
Review : {review}
""")

prompt=template.format({
    "name" : "Altammash",
    "age" : "21",
    "review" : "I am a good boy."
})

llm = NakliLLM()

llm.predict(prompt)

class NakliLLMChain():
    def __init__(self,llm,template):
        self.llm = llm 
        self.template = template

    def run(self,dict):
        final_prompt= self.prompt.format(dict)
        return self.llm.predict(final_prompt)

llm_chain = NakliLLMChain(llm,template)
llm_chain.run({
    "name" : "Altammash",
    "age" : "21",
    "review" : "I am a good boy."
})


class RunnableConnector(Runnable):
    def __init__ (self,runnable_list):
        self.runnable_list=runnable_list
    def invoke(self,invoke_data):

        for runnable in self.runnable_list:
            invoke_data = runnable.invoke(invoke_data)
        return invoke_data

runnable_connector = RunnableConnector([template,llm])
result = runnable_connector.invoke({
    "name" : "Altammash",
    "age" : "21",
    "review" : "I am a good boy."
})
print(result)