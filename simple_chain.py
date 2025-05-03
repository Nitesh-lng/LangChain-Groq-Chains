from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm=ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct"
)
prompt =PromptTemplate(
    template='tell me 1 interesting fact about {input}',
    input_variables=["input"]
)

parser=StrOutputParser()

chain= prompt | llm | parser

result = chain.invoke({"input": "steve jobs"})

print(result)

chain.get_graph().print_ascii()