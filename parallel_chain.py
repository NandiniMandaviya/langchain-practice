from dotenv import load_dotenv
import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

prompt1 = PromptTemplate(
    template = "Generate a tweet about {topic} in a fun way.",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
   template = "Generate a linkedin post about {topic} in a professional tone.",
   input_variables= ['text'] 
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = RunnableParallel({
    'tweet' : prompt1 | model | parser,
    'linkedin' : prompt2 | model | parser
})

result = chain.invoke({'topic' : 'AI'})

print("Tweet: \n" + result['tweet'])
print("LinkedIn Post: \n" + result['linkedin'])
