from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    task="text-generation",
   
)
model1= ChatGoogleGenerativeAI(model='gemini-2.5-flash')
model2 = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a notes from the given text \n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 quizes from the given text\n{text}',
    input_variables=['text']
)

prompt3= PromptTemplate(
    template="Merge the notes and quizes into a single document.First section of document should contain notes and second section should contain quizes. \n notes->{notes} \n quiz->{quiz}",
    input_variables=['notes','quiz']
)

paraller_chain = RunnableParallel(
    {
        'notes': prompt1 | model1 | parser,
        'quiz': prompt2 | model2 | parser
    }
)

merge_chain= prompt3 | model1 | parser

final_chain = paraller_chain | merge_chain

text=""" Elizabeth Lyon (fl. c. 1722–1726), nicknamed Edgworth Bess or Edgware Bess, was an English thief, a prostitute, and the partner of the criminal Jack Sheppard. Little is known about her background or her early life, but it is known that she was working as a prostitute at the Black Lyon alehouse in London by 1722 or 1723. Here she met Sheppard—at the time an apprentice carpenter—and the two began a relationship.

At Lyon's instigation, Sheppard soon began his career in crime, first stealing from places where he worked before moving into housebreaking; Lyon and his brother became his accomplices. Sheppard was arrested for his crimes on several occasions, invariably breaking out soon after incarceration, normally assisted by Lyon. In May 1724 she was arrested when visiting him in prison, so together the pair broke out of New Prison into the adjoining Clerkenwell Bridewell prison, then out of that to freedom.

After Sheppard's execution in November 1724, Lyon entered into relationships with other men who were, or became, involved in housebreaking, sometimes accompanying them to assist in perpetrating the crimes. She was arrested in March 1726 and transported to the Province of Maryland—then a British colony in North America—in October; her name does not appear in official sources after that date. Lyon's notoriety is based on her connection to Sheppard, and in the years following his execution, novels were published and plays performed that retold his and Lyon's story.  """

result = final_chain.invoke({'text':text})
print(result)

final_chain.get_graph().print_ascii()
