import os
import requests
from dotenv import load_dotenv
from typing import Annotated

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain.agents import create_agent

load_dotenv()

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    model="DeepSeek-V4-Pro",
)

search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke("What is the top 5 news in India today?")
print(results)

response = model.invoke('Hi!')
print(response)

agent = create_agent(
    model=model,
    tools=[search_tool],
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "Three ways to reach Goa from Delhi."
            )
        }
    ]
})

print(response)
