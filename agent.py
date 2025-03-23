import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Verify Google API key is present
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please add it to your .env file.")

async def main():
    # Instantiate the Gemini chat model with explicit API key
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        temperature=0.7,
        max_tokens=512,
        max_retries=2,
        google_api_key=os.getenv("GOOGLE_API_KEY")  # Explicitly set the API key
    )

    # Connect to your MCP servers using the MultiServerMCPClient
    async with MultiServerMCPClient(
        {
            "math": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    ) as client:
        tools = client.get_tools()
        agent = create_react_agent(model, tools)
        
        # Invoke the agent with a prompt that calls the "add" tool
        math_response = await agent.ainvoke({
            "messages": [{"role": "user", "content": "What is 2+2?"}]
        })
        print("Math Response:", math_response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())