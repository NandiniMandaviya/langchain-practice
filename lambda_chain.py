from dotenv import load_dotenv
import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

prompt1 = PromptTemplate(
    template = "Write a joke about {topic}.",
    input_variables = ['topic']
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

joke_generator = prompt1 | model | parser

length = RunnableLambda(lambda x: len(x.split()))

parallel_chain = RunnableParallel({  
    'joke' : RunnablePassthrough(),
    'length' : length
})

chain = joke_generator | parallel_chain

result = chain.invoke({'topic': 'AI'})

print("The joke is as- \n" + result['joke'])
print("The word length of the joke is- \n" + str(result['length']))
