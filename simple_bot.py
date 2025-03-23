import discord
from discord.ext import commands
import asyncio
import sys
import signal
import json
import os
import pickle
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.schema import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Memory imports
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore

# Load environment variables
load_dotenv(find_dotenv())

print("MISTRAL_API_KEY:", os.environ.get("MISTRAL_API_KEY"))

# Verify required API keys are present
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
if "MISTRAL_API_KEY" not in os.environ:
    raise ValueError("MISTRAL_API_KEY environment variable not set")

# Discord setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Setup memory directory for simple persistence
memory_dir = "./memory_data"
os.makedirs(memory_dir, exist_ok=True)
memory_file = os.path.join(memory_dir, "memory_store.pkl")

# Initialize memory store
memory_store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "mistralai:mistral-embed",
    }
)

# Try to load previous memories if they exist
try:
    if os.path.exists(memory_file):
        with open(memory_file, "rb") as f:
            loaded_store = pickle.load(f)
            memory_store = loaded_store
            print("Successfully loaded memory from disk")
except Exception as e:
    print(f"Could not load memory from disk, starting fresh: {e}")

# Setup memory tools
memory_tools = [
    create_manage_memory_tool(namespace=("memories",)),
    create_search_memory_tool(namespace=("memories",))
]

# Initialize memory agent
memory_agent = None

# Function to save memory to disk
def save_memory_to_disk():
    try:
        with open(memory_file, "wb") as f:
            pickle.dump(memory_store, f)
        print("Memory saved to disk")
    except Exception as e:
        print(f"Error saving memory to disk: {e}")

# Async version of save_memory_to_disk
async def save_memory_to_disk_async():
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, save_memory_to_disk)
    except Exception as e:
        print(f"Error in save_memory_to_disk_async: {e}")

# Memory prompt function with timeout protection
def memory_prompt(state):
    """Add memory context to the prompt with timeout protection"""
    last_message = state["messages"][-1].content if state["messages"] else ""
    try:
        # Use a very short timeout for memory operations to prevent heartbeat blocking
        memory_text = "Unable to access detailed memory at this time."
        
        # Only do a very basic search (limit results and complexity)
        memories = memory_store.search(("memories",), query=last_message, limit=2)
        if memories:
            memory_text = "\n".join([mem.value["content"] for mem in memories])
    except Exception as e:
        print(f"Error in memory prompt: {e}")
        memory_text = "Memory access limited to prevent timeouts."
    
    # Create system message with memory context but keep it lightweight
    system_content = "Memory context: " + memory_text + "\nYou are a helpful Discord bot."
    
    # Return the messages with minimal processing
    return [{"role": "system", "content": system_content}] + state["messages"]

# Add memory entry safely (without blocking)
async def add_memory_entry(question, response, key_prefix="memory"):
    try:
        # Create a unique key
        memory_key = f"{key_prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create a simpler memory entry - keep it lightweight
        memory_entry = {
            "messages": [
                {"role": "user", "content": question}
            ],
            "namespace": ("memories",),
            "key": memory_key,
            "value": {
                "content": "User: " + question[:100] + "... Bot: " + response[:100] + "..."
            }
        }
        
        # Use a very short timeout
        await asyncio.wait_for(
            memory_agent.ainvoke(memory_entry, config={"configurable": {"thread_id": "memory"}}),
            timeout=0.5  # Ultra short timeout to prevent blocking
        )
        return True
    except Exception as e:
        print(f"Error adding memory: {e}")
        return False

# Load and process documents
loader = TextLoader("./maitreya.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Mistral AI Embeddings
embeddings = MistralAIEmbeddings(model="mistral-embed")
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
vectorstore.persist()  # Ensure data is saved to disk
retriever = vectorstore.as_retriever()

# Google Chat model
chat = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.0
)

# Get relevant context from vector database - with timeout protection
async def get_context(question):
    try:
        # Set a timeout for retriever to prevent blocking Discord
        retrieval_task = asyncio.create_task(
            asyncio.wait_for(
                retriever.ainvoke(question),
                timeout=3.0  # 3-second timeout
            )
        )
        docs = await retrieval_task
        if docs:
            return "\n\n".join([doc.page_content for doc in docs])
    except asyncio.TimeoutError:
        print("Context retrieval timed out")
        return "Context retrieval timed out, please try again."
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return f"Error retrieving context: {e}"
    return "No relevant information found in the knowledge base."

# Get calculation results using MCP tools
async def get_calculation(question):
    try:
        # Connect to the MCP server with the correct configuration
        async with MultiServerMCPClient(
            {
                "math": {
                    "url": "http://localhost:8000/sse",
                    "transport": "sse",
                },
                "webcrawl": {
                    "url": "http://localhost:8001/sse",
                    "transport": "sse",
                }
            }
        ) as mcp_client:
            # Get tools from the MCP server
            tools = mcp_client.get_tools()
            
            # Create a temporary calculation agent
            calc_agent = create_react_agent(
                chat,  # Use our existing chat model
                tools=tools
            )
            
            # Use a more focused question for calculation
            calc_question = "Calculate this: " + question
            
            # Invoke the agent with the calculation question
            response = await calc_agent.ainvoke(
                {"messages": [{"role": "user", "content": calc_question}]},
                config={"configurable": {"thread_id": "calc-thread"}}
            )
            
            # Extract the result
            return response["messages"][-1].content
    except Exception as e:
        print(f"Calculation error: {e}")
        return f"I couldn't perform that calculation: {str(e)}"

# Combine context, calculation, and memory for a comprehensive response
async def get_combined_response(question):
    try:
        # Get context from vector DB with a short timeout
        context = await asyncio.wait_for(get_context(question), timeout=2.0)
        
        # Extract any potential calculations from the question
        math_keywords = ["calculate", "what is", "compute", "sum", "add", "subtract", 
                        "multiply", "divide", "+", "-", "*", "/", "="]
        
        calculation_result = None
        if any(keyword in question.lower() for keyword in math_keywords):
            # Get calculation with short timeout
            calculation_result = await asyncio.wait_for(get_calculation(question), timeout=2.0)
        
        # Build a simple system message
        system_content = "You are a helpful Discord bot."
        if context:
            system_content += " Here's what I know: " + context
        if calculation_result:
            system_content += " Calculation result: " + calculation_result
        
        # Create minimal messages for faster processing
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question}
        ]
        
        # Use the memory agent with memory_prompt, but with a strict timeout
        try:
            # Use a thread_id for potential tracking, but keep it simple
            config = {"configurable": {"thread_id": "chat"}}
            
            # Invoke with a strict timeout to prevent heartbeat issues
            response = await asyncio.wait_for(
                memory_agent.ainvoke({"messages": messages}, config=config),
                timeout=3.0  # 3-second strict timeout
            )
            
            # Extract the response content
            if "messages" in response and response["messages"]:
                return response["messages"][-1].content
            else:
                return str(response)
        except asyncio.TimeoutError:
            # Fast fallback if memory agent times out
            if context:
                return "Based on what I know: " + context
            else:
                return "I don't have specific information about that."
        except Exception as e:
            print(f"Error using memory agent: {e}")
            # Simple fallback
            if context:
                return "I found this relevant information: " + context
            else:
                return "I don't have information about that right now."
    except Exception as e:
        print(f"Error in combined response: {e}")
        return "I'm sorry, I encountered an error. Please try a simpler question."

@bot.command()
async def power(ctx, *, question=None):
    """Command to ask a question with memory, context and calculation capabilities"""
    if not question:
        await ctx.send("Please provide a question after !power")
        return
    
    try:
        # Start typing indicator
        async with ctx.typing():
            # Set a timeout for the entire operation
            response = await asyncio.wait_for(
                get_combined_response(question),
                timeout=8.0  # Enough time for most operations, but not too long
            )
            
            # Send the response first (don't make user wait for memory)
            await ctx.send(response)
            
            # Then handle memory in the background
            asyncio.create_task(add_memory_entry(question, response, key_prefix="power"))
            
            # Occasionally save to disk
            if hash(str(ctx.message.id)) % 3 == 0:
                asyncio.create_task(save_memory_to_disk_async())
                
    except asyncio.TimeoutError:
        await ctx.send("I'm sorry, but your request took too long to process. Could you try a simpler question?")
    except Exception as e:
        print(f"Error in power command: {e}")
        await ctx.send(f"I'm sorry, I encountered an error processing your request. Please try again.")

@bot.event
async def on_ready():
    global memory_agent
    
    print(f'{bot.user} has connected to Discord!')
    
    # Create the memory agent with simple InMemoryStore
    memory_agent = create_react_agent(
        chat,
        tools=memory_tools,
        prompt=memory_prompt
    )
    
    print("Memory agent created successfully")
    print("Bot is ready to use with command !power")

# Listen to messages in the channel
@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return
    
    # Only respond to messages that tag the bot or are direct messages
    if bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
        question = message.content
        
        # Remove bot mention from question
        for mention in message.mentions:
            question = question.replace(f'<@{mention.id}>', '').replace(f'<@!{mention.id}>', '')
        
        question = question.strip()
        
        # Generate a typing indicator while processing
        async with message.channel.typing():
            # Get combined response with memory, calculation, and context
            response = await get_combined_response(question)
            
            # Send the response first (don't make user wait for memory operations)
            await message.channel.send(response)
            
            # Then save memory in the background without blocking
            asyncio.create_task(add_memory_entry(question, response))
            
            # Occasionally save to disk (use a fast hashing trick to save only some of the time)
            if hash(str(message.id)) % 5 == 0:
                # Use create_task so we don't block
                asyncio.create_task(save_memory_to_disk_async())
    
    # Process commands like !power
    await bot.process_commands(message)

# Modified signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("Shutting down bot...")
    
    # Save vector store
    vectorstore.persist()
    print("Vector store persisted")
    
    # Save memory to disk
    save_memory_to_disk()
    print("Memory saved to disk")
    
    # Exit without trying to access event loop
    sys.exit(0)

# Register signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)

# Run the bot
if __name__ == "__main__":
    if not os.environ.get("DISCORD_TOKEN"):
        raise ValueError("DISCORD_TOKEN environment variable not set")
    
    # Start the bot
    bot.run(os.environ.get("DISCORD_TOKEN"))