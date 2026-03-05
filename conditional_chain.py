from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
   
)
model1= ChatHuggingFace(llm=llm)
parser = StrOutputParser()

class Sentiment(BaseModel):
    sentiment : Literal['positive','negative']=Field(description='Give the Sentiment of the feedback')

pydantic_parser= PydanticOutputParser(pydantic_object=Sentiment)

prompt1= PromptTemplate(
    template="Generate the sentiment of given feedback {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions':pydantic_parser.get_format_instructions()}
)

classifier_chain = prompt1 | model1 | pydantic_parser


prompt2= PromptTemplate(
    template="Generate a response for the given feedback if sentiment is positive {feedback}",
    input_variables=['feedback']
)

prompt3= PromptTemplate(
    template="Generate a response for the given feedback if sentiment is negative {feedback}",
    input_variables=['feedback']
)

branch_chain=RunnableBranch(
    (lambda x: x.sentiment == 'positive',prompt2 | model1 | parser),
    (lambda x: x.sentiment == 'negative',prompt3 | model1 | parser),
    RunnableLambda(lambda x:"Invalid sentiment")
)


final_chain = classifier_chain | branch_chain

result = final_chain.invoke({'feedback':'The phone is dreadful'})

print(result)
