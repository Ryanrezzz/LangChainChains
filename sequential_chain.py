from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser= StrOutputParser()

prompt1 = PromptTemplate(
    template='Explain about this {topic} ',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the given {text} in 4 lines',
    input_variables=['text']
)

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Lamine Yamal'})

print(result)
chain.get_graph().print_ascii()