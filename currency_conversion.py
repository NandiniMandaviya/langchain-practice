import os
import requests
from dotenv import load_dotenv
from typing import Annotated

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg
from langchain.tools import tool

load_dotenv()

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    model="DeepSeek-V4-Pro",
)


@tool
def get_conversion_factor(
    base_currency: str,
    target_currency: str
) -> float:
    """
    Fetch the conversion factor between two currencies.
    """

    url = (
        f"https://v6.exchangerate-api.com/v6/"
        f"d435222483828a16a48588a1/"
        f"pair/{base_currency}/{target_currency}"
    )

    response = requests.get(url)
    response.raise_for_status()

    return response.json()["conversion_rate"]


@tool
def convert_to_currency(
    base_currency_value: float,
    conversion_factor: Annotated[float, InjectedToolArg]
) -> float:
    """
    Convert a currency value using the given conversion factor.
    """

    return base_currency_value * conversion_factor


tools = {
    "get_conversion_factor": get_conversion_factor,
    "convert_to_currency": convert_to_currency
}


llm_with_tools = model.bind_tools([
    get_conversion_factor,
    convert_to_currency
])


messages = [
    HumanMessage(
        content=(
            "What is the conversion factor between USD and INR, "
            "and based on that, can you convert 10 USD to INR?"
        )
    )
]


conversion_factor = None

while True:

    ai_message = llm_with_tools.invoke(messages)

    # Add the AI response to conversation history
    messages.append(ai_message)


    if not ai_message.tool_calls:
        print("\nFINAL ANSWER:")
        print(ai_message.content)
        break

    for tool_call in ai_message.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"\nCalling tool: {tool_name}")
        print(f"Arguments: {tool_args}")

        selected_tool = tools[tool_name]

        if tool_name == "convert_to_currency":
            tool_args["conversion_factor"] = conversion_factor

        tool_result = selected_tool.invoke(tool_args)

        print(f"Tool result: {tool_result}")

        if tool_name == "get_conversion_factor":
            conversion_factor = tool_result

        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )