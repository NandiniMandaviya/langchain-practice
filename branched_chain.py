from dotenv import load_dotenv
import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

prompt1 = PromptTemplate(
    template = "Write an report on the following topic: {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template = "Summarize the following report \n: {text}",
    input_variables = ['text']
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

report_generator = prompt1 | model | parser

branched_chain = RunnableBranch(
    (lambda x: 'artificial' in x.lower().split() , prompt2 | model | parser),
    RunnablePassthrough()
)

chain = report_generator | branched_chain

result = chain.invoke({'topic': 'AI'})

print(result)

chain.get_graph().print_ascii()