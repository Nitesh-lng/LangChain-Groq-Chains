from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm1=ChatGroq(
    model='meta-llama/llama-4-scout-17b-16e-instruct'
)

llm2=ChatGroq(
    model='deepseek-r1-distill-llama-70b'
)

llm3=ChatGroq(
    model='deepseek-r1-distill-llama-70b'
)

prompt1=PromptTemplate(
    template='create notes in 200 words of{text}',
    input_variables=['text']
)
prompt2=PromptTemplate(
    template='create quiz of 5 questions from {text}',
    input_variables=['text']
)
prompt3=PromptTemplate(
    template='merge the both quiz->{quiz} and notes->{notes}',
    input_variables=['quiz','notes']
)

parser=StrOutputParser()

parellel_chain=RunnableParallel(
    {
        "notes": prompt1 | llm1 | parser,
        "quiz": prompt2 | llm2 | parser, 
    }
)

merge_chain=prompt3 | llm3 | parser
chain =parellel_chain | merge_chain


text=""" 
     THE ELEVENTH APARTMENT had only one closet, but it did have a sliding glass door that opened onto a small balcony, from which he could see a man sitting across the way, outdoors in only a T-shirt and shorts even though it was October, smoking. Willem held up a hand in greeting to him, but the man didnt wave back.
In the bedroom, Jude was accordioning the closet door, opening and shutting it, when Willem came in. “Theres only one closet,” he said.
“Thats okay,” Willem said. “I have nothing to put in it anyway.”
“Neither do I.” They smiled at each other. The agent from the building wandered in after them. “Well take it,” Jude told her.
But back at the agents office, they were told they couldnt rent the apartment after all. “Why not?” Jude asked her.
“You dont make enough to cover six months rent, and you dont have anything in savings,” said the agent, suddenly terse. She had checked their credit and their bank accounts and had at last realized that there was something amiss about two men in their twenties who were not a couple and yet were trying to rent a one-bedroom apartment on a dull (but still expensive) stretch of Twenty-fifth Street. “Do you have anyone who can sign on as your guarantor? A boss? Parents?”
“Our parents are dead,” said Willem, swiftly.
The agent sighed. “Then I suggest you lower your expectations. No one who manages a well-run building is going to rent to candidates with your financial profile.” And then she stood, with an air of finality, and looked pointedly at the door.
When they told JB and Malcolm this, however, they made it into a comedy: the apartment floor became tattooed with mouse droppings, the man across the way had almost exposed himself, the agent was upset because she had been flirting with Willem and he hadnt reciprocated.
"""
result = chain.invoke({"text":text})

print(result)

chain.get_graph().print_ascii()