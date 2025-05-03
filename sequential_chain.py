from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os 

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm=ChatGroq(
    model='meta-llama/llama-4-scout-17b-16e-instruct'
)

prompt1=PromptTemplate(
    template='details about {input}',
    input_variable=['input']
)

prompt2=PromptTemplate(
    template='give me 2 lines summary of {text}',
    input_variable=['text']
)

parser=StrOutputParser()

chain=prompt1 | llm | parser | prompt2 | llm | parser

result =chain.invoke({'input':'HB1 visa'})
print(result)