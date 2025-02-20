import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
 

load_dotenv()

class GroqAnswering:    
    def __init__(self, cont, query):
        self.cont = cont
        self.query = query
        self.chat = ChatGroq(temperature=0, 
                             groq_api_key=os.getenv("GROQ_API_KEY"), 
                             model_name="llama-3.3-70b-versatile")
        self.system = "You are a helpful agent, which give answere only based on provided context. Don't use your own knowledge"

        
    def groq_answering(self,):    

        prompt = ChatPromptTemplate.from_messages([("system", self.system), ("human", "Context:\n{context}\n\n Question:{query}")])
        chain = prompt | self.chat
        request = chain.invoke({"context": self.cont, "query":self.query})

        return request.content