import asyncio
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()


async def main():
    # Connect to MCP servers
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

    # Load tools from both servers
    tools = await client.get_tools()

    # Load LLM
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
    )

    # Create ReAct agent
    agent = create_react_agent(
        model=model,
        tools=tools,
    )

    # Ask a question
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Chennai?"
                }
            ]
        }
    )

    # Print final response
    print("\nAssistant:")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())