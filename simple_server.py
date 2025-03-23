from mcp.server.fastmcp import FastMCP

# Create an MCP server named "Math"
mcp = FastMCP("Math")

mcp.settings.port = 8000

# Expose the add tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    # Run the server using stdio transport so that it works with the LangChain client
    mcp.run(transport="sse")