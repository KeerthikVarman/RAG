from fastmcp import FastMCP

mcp=FastMCP("weather")


@mcp.tool()
async def get_weather(city: str) -> str:
    """get the weather forecast for a city"""
    return f"weather in {city} is sunny"

if __name__=="__main__":
    mcp.run(transport="streamable-http")
