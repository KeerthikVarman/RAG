import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()


async def main():
    client = MultiServerMCPClient(
        {
            "maths": {
                "command": "python",
                "args": ["mathsserver.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://127.0.0.1:8000/mcp",
                "transport": "streamable-http",
            },
        }
    )

    tools = await client.get_tools()

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
    )

    agent = create_react_agent(
        model=model,
        tools=tools,
    )

    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is addition of 3+5."
                }
            ]
        }
    )
    print("\nAssistant:")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())