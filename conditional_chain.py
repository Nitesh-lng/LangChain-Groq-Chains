from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel , RunnableBranch , RunnableLambda
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm=ChatGroq(
    model='llama3-8b-8192'
)

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"]= Field(description="sentiment of the text")
    
parser1=StrOutputParser()
parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt=PromptTemplate(
    template='classify the sentiment of the given feedback text into positive or negative. {text} \n {format_instructions}',
    input_variables=['text'],
    partial_variables={
        "format_instructions": parser2.get_format_instructions()
    }
)

classifier_chain= prompt | llm | parser2

prompt2=PromptTemplate(
    template='write a appropriate of this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='write a appropriate of this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain=RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | llm | parser1),
    (lambda x: x.sentiment == 'negative', prompt3 | llm | parser1),
    RunnableLambda(lambda x: "could not classify the feedback")
)

chain=classifier_chain | branch_chain

text="Your work presented a compellingly ambiguous resonance, a paradoxically resolved tension akin to a pre-dream memory. The subject's presence felt both familiar and alien, like a distorted existential reflection. Intentional inconsistencies fostered a delightful bewilderment, making the expected predictably unexpected. A sense of significant understanding lingered, yet its precise nature remained gloriously out of reach. Congratulations, maybe? The answer, like this feedback, is wonderfully unclear."

result=chain.invoke({"text":text})

print(result)