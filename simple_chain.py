from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser= StrOutputParser()

prompt = PromptTemplate(
    template='tell me about {topic} in 5 points.',
    input_variables=['topic']
)

chain = prompt | model | parser

result = chain.invoke({'topic':'Lamine Yamal'})

print(result)
chain.get_graph().print_ascii()