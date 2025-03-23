import discord
from discord.ext import commands
import asyncio
from dotenv import load_dotenv, find_dotenv
import os
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.schema import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import re
import io
import datetime as dt
import dateparser

from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv(find_dotenv())
print("[OldBot] Environment variables loaded.")



if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
if "MISTRAL_API_KEY" not in os.environ:
    raise ValueError("MISTRAL_API_KEY environment variable not set")


intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)
print("[OldBot] Discord bot initialized.")


loader = TextLoader("./maitreya.txt")
documents = loader.load()
print("[OldBot] Documents loaded.")
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
print(f"[OldBot] Documents split into {len(texts)} chunks.")


embeddings = MistralAIEmbeddings(model="mistral-embed")
retriever = Chroma.from_documents(texts, embeddings).as_retriever()
print("[OldBot] Vector store created with document embeddings.")


chat = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.0
)
print("[OldBot] Chat model initialized.")

memory_store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "mistralai:mistral-embed",
    }
)
memory_tools = [
    create_manage_memory_tool(namespace=("memories",)),
    create_search_memory_tool(namespace=("memories",))
]
print("[OldBot] Memory store and tools set up.")

def memory_prompt(state):
    last_message = state["messages"][-1].content if state["messages"] else ""
    print(f"[OldBot] Building memory prompt. Last message: {last_message}")
    memories = memory_store.search(("memories",), query=last_message)
    memory_text = "\n\n".join([mem.value["content"] for mem in memories]) if memories else "No past memory."
    system_msg = {"role": "system", "content": f"Memory from past conversations:\n{memory_text}"}
    return [system_msg] + state["messages"]


memory_agent = create_react_agent(
    chat,
    prompt=memory_prompt,
    tools=memory_tools,
    store=memory_store,
    checkpointer=InMemorySaver()
)
print("[OldBot] Memory agent created.")


async def get_context(question):
    print(f"[OldBot] Retrieving context for question: {question}")
    docs = retriever.get_relevant_documents(query=question)
    if docs:
        context = "\n\n".join([doc.page_content for doc in docs])
        print("[OldBot] Context retrieved.")
        return context
    print("[OldBot] No relevant context found.")
    return "No relevant information found in the knowledge base."

# Handle sending potentially long responses
async def send_response(ctx, content, force_markdown=False):
    if force_markdown or len(content) > 2000:
        file = discord.File(
            fp=io.BytesIO(content.encode('utf-8')),
            filename="response.md"
        )
        await ctx.send("The response is too long for Discord. Here's a markdown file:", file=file)
    else:
        # Split content into chunks of 2000 characters
        for i in range(0, len(content), 2000):
            chunk = content[i:i+2000]
            await ctx.send(chunk)


async def get_mcp_result(question):
    try:
        print(f"[OldBot] Initiating MCP tool call for question: {question}")
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            temperature=0.0,
            max_tokens=512,
            max_retries=2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        
        mcp_servers = {
            "webcrawl": {
                "url": "http://localhost:8001/sse",
                "transport": "sse",
                "enabled": True,  
                "keywords": [
                    "scrape", "crawl", "web", "extract", "data", "website", "page", "content",
                    "search", "find", "query", "text matching", "analyze", "understand", 
                    "summarize", "smart extract", "intelligently"
                ]
            },
            "gmail": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
                "enabled": True,  
                "keywords": [
                    "email", "gmail", "mail", "inbox", "message", "fetch", "recent emails",
                    "send email", "check email", "read email", "unread", "folder"
                ]
            }
        }
        
        
        request_type = None
        
        
        if any(keyword in question.lower() for keyword in ["email", "gmail", "mail", "inbox"]):
            if mcp_servers["gmail"]["enabled"]:
                request_type = "gmail"
                print(f"[OldBot] Detected email request: {question}")
        
        
        if not request_type:
            
            urls = re.findall(r'https?://[^\s]+', question)
            
            
            is_web_request = any(keyword in question.lower() for keyword in mcp_servers["webcrawl"]["keywords"])
            
            if urls and is_web_request and mcp_servers["webcrawl"]["enabled"]:
                request_type = "webcrawl"
                print(f"[OldBot] Detected web extraction request: {question}")
        
        
        if not request_type:
            print(f"[OldBot] No matching MCP server for question: {question}")
            return None
        
        
        active_servers = {}
        if request_type == "gmail" and mcp_servers["gmail"]["enabled"]:
            active_servers["gmail"] = {
                "url": mcp_servers["gmail"]["url"],
                "transport": mcp_servers["gmail"]["transport"]
            }
        elif request_type == "webcrawl" and mcp_servers["webcrawl"]["enabled"]:
            active_servers["webcrawl"] = {
                "url": mcp_servers["webcrawl"]["url"],
                "transport": mcp_servers["webcrawl"]["transport"]
            }
        
        
        if not active_servers:
            print("[OldBot] No active MCP servers configured for this request")
            return None
            
       
        try:
            async with MultiServerMCPClient(active_servers) as client:
                tools = client.get_tools()
                print(f"[OldBot] Available tools: {[t.name for t in tools]}")
                
            
                if request_type == "gmail":
                    return await handle_email_request(question, tools)
                
                
                elif request_type == "webcrawl":
                    return await handle_web_extraction(question, tools, urls[0] if urls else None)
                
        except Exception as connection_error:
            error_msg = str(connection_error)
            print(f"[OldBot] MCP server connection error: {error_msg}")
            
            if request_type == "gmail":
                return f"Error connecting to Gmail MCP server: {error_msg}. Please ensure it's running on port 8000 and your .env file is configured with SMTP_USERNAME and SMTP_PASSWORD."
            else:
                return f"Error connecting to Web Crawling MCP server: {error_msg}. Please ensure it's running on port 8001."
                
    except Exception as e:
        print(f"[OldBot] MCP tool call error: {str(e)}")
        return f"Could not perform MCP tool call: {str(e)}"

# Handler for email-related requests
async def handle_email_request(question, tools):
    """Process email-related requests using the Gmail MCP server"""
    print(f"[OldBot] Processing email request: {question}")
    
    # Check if this is a scheduling request
    is_schedule_request = False
    
    # Only consider it a scheduling request if specific future scheduling keywords are present
    # AND "right now" or "immediately" are NOT present
    scheduling_keywords = ["schedule", "later", "future", "remind", "send at", "send on"]
    immediate_keywords = ["right now", "immediately", "now", "right away", "instantly", "asap"]
    
    has_scheduling_words = any(keyword in question.lower() for keyword in scheduling_keywords)
    has_immediate_words = any(keyword in question.lower() for keyword in immediate_keywords)
    
    # If it has scheduling keywords but also immediate keywords, prioritize immediate sending
    if has_scheduling_words and not has_immediate_words:
        # Check for time pattern that looks like scheduling
        time_pattern = re.search(r'(at|on)\s+([\w\s\d,:.-]+)', question, re.IGNORECASE)
        if time_pattern:
            is_schedule_request = True
    
    if is_schedule_request:
        # Try to extract a date/time from the query
        schedule_match = re.search(r'(send|schedule).*?(on|at|for)\s+([\w\s\d,:.-]+)', question, re.IGNORECASE)
        schedule_time = None
        
        if schedule_match:
            time_str = schedule_match.group(3).strip()
            try:
                # Attempt to parse the date-time string
                # First try with specific format
                try:
                    schedule_time = dt.datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                except ValueError:
                    # Try with more natural language - dateparser handles am/pm correctly
                    schedule_time = dateparser.parse(time_str, settings={'PREFER_DATES_FROM': 'future'})
                    
                if schedule_time and schedule_time > dt.datetime.now():
                    # Format the time for the API
                    schedule_time_iso = schedule_time.isoformat()
                    
                    # Extract email details
                    parts = question.split("schedule", 1)
                    if len(parts) < 2:
                        parts = question.split("later", 1)
                    if len(parts) < 2:
                        parts = question.split("remind", 1)
                    
                    if len(parts) >= 2:
                        email_query = parts[1].split(schedule_match.group(0))[0] + parts[1].split(schedule_match.group(0))[1]
                    else:
                        email_query = question
                    
                    # Extract essential email information (recipient, subject, body)
                    recipient_match = re.search(r'to\s+([^\s]+@[^\s]+)', email_query)
                    subject_match = re.search(r'subject\s+["\'](.+?)["\']', email_query)
                    body_match = re.search(r'body\s+["\'](.+?)["\']', email_query)
                    
                    recipient = recipient_match.group(1) if recipient_match else None
                    subject = subject_match.group(1) if subject_match else "Scheduled Email"
                    body = body_match.group(1) if body_match else "This is a scheduled email."
                    
                    if not recipient:
                        return "I couldn't determine who to send the email to. Please include 'to recipient@example.com' in your request."
                    
                    # Call the schedule_email tool
                    try:
                        result = client.invoke_tool("gmail", "schedule_email", {
                            "recipient": recipient,
                            "subject": subject,
                            "body": body,
                            "schedule_time": schedule_time_iso
                        })
                        return f"Email scheduled for {schedule_time.strftime('%Y-%m-%d %H:%M')}\n\n{result}"
                    except Exception as e:
                        return f"Error scheduling email: {str(e)}"
            except Exception as parse_error:
                return f"I couldn't understand the scheduling time: {str(parse_error)}\nPlease use a format like 'YYYY-MM-DD HH:MM' or a natural language time description."
    else:
        # For other email operations, let the LLM handle it with all email tools
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            temperature=0.0,
            max_tokens=512,
            max_retries=2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Find email-related tools
        email_tools = [tool for tool in tools if tool.name in ["fetch_recent_emails", "send_email_tool"]]
        
        if email_tools:
            agent = create_react_agent(model, email_tools)
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": "email-thread"}}
            )
            print("[OldBot] Email agent returned a response.")
            return response["messages"][-1].content
        else:
            return "Gmail tools are not available. Please make sure the Gmail MCP server is running."

# Updated handler for web extraction requests
async def handle_web_extraction(question, tools, url=None):
    """Process web extraction requests using web crawling tools"""
    print(f"[OldBot] Processing web extraction request: {question}")
    
    # If URL is not provided, try to extract it from the question
    if not url:
        urls = re.findall(r'https?://[^\s]+', question)
        if urls:
            url = urls[0]
        else:
            return "No URL found in your request. Please include a valid URL."
    
    # Get available tool names
    tool_names = [t.name for t in tools]
    
    # Check for various types of web extraction requests
    is_scrape_request = any(keyword in question.lower() for keyword in 
                           ["scrape", "crawl", "web", "extract", "data", "website", "page", "content"])
    is_search_request = any(keyword in question.lower() for keyword in 
                           ["find", "search", "look for", "locate", "query", "text matching"])
    is_smart_request = any(phrase in question.lower() for phrase in 
                           ["analyze", "understand", "explain", "extract information", "smart extract", 
                            "intelligently", "get info about", "tell me about", "summarize"])
    
    # Select the appropriate tool based on the request type and available tools
    tool_to_use = None
    invoke_params = {"url": url}
    
    # Determine which tool to use based on the request type
    if is_smart_request and "smart_extract" in tool_names:
        tool_name = "smart_extract"
        # Try to extract an instruction from the question
        instruction_patterns = [
            (r'(?:tell|give) me about ["\'](.*?)["\']', 1),
            (r'(?:extract|find|get) (?:info|information) (?:about|on|regarding) ["\'](.*?)["\']', 1),
            (r'(?:extract|find|get) (?:info|information) (?:about|on|regarding) (.*?)(?:from|on|in)', 1),
            (r'(?:summarize|analyze) (.*?)(?:from|on|in)', 1),
            (r'(?:extract information about) ["\'](.*?)["\']', 1)
        ]
        
        for pattern, group in instruction_patterns:
            instruction_match = re.search(pattern, question, re.IGNORECASE)
            if instruction_match:
                invoke_params["instruction"] = instruction_match.group(group).strip()
                break
        
       
        if "instruction" not in invoke_params:
            # Clean up the question to formulate an instruction
            clean_question = re.sub(r'(?:could you|can you|please|for me).*?', '', question.lower())
            clean_question = re.sub(r'(?:https?://[^\s]+)', '', clean_question)
            invoke_params["instruction"] = f"Extract information about {clean_question.strip()}"
    
    elif is_search_request and "extract_text_by_query" in tool_names:
        tool_name = "extract_text_by_query"
        # Look for query terms
        query_patterns = [
            (r'(search|find|query|look) for ["\']([^"\']+)["\']', 2),
            (r'(search|find|query|look) for (.+?)(in|on|from|using)', 2),
            (r'containing ["\']([^"\']+)["\']', 1),
            (r'about ["\']([^"\']+)["\']', 1)
        ]
        
        for pattern, group in query_patterns:
            query_match = re.search(pattern, question.lower())
            if query_match:
                invoke_params["query"] = query_match.group(group).strip()
                break
        
        
        if "query" not in invoke_params:
            
            cleaned_question = re.sub(r'(search|find|extract|get|retrieve|look for|tell me about|information on|data about).*?(on|in|from|at|using).*?https?://[^\s]+', '', question.lower())
            words = cleaned_question.split()
            if len(words) > 2:  # Avoid single words as queries
                invoke_params["query"] = " ".join(words[-3:])  # Use last few words as query
            else:
                
                tool_name = "scrape_url"
                invoke_params = {"url": url}
    
    else:
        tool_name = "scrape_url"
    
    
    for tool in tools:
        if tool.name == tool_name:
            tool_to_use = tool
            break
    
    if tool_to_use:
        print(f"[OldBot] Directly invoking {tool_name} tool with params: {invoke_params}")
        try:
            content = await tool_to_use.ainvoke(invoke_params)
            print(f"[OldBot] Successfully retrieved content, length: {len(content)}")
            
            # Check if content is very long
            if len(content) > 4000:
                return f"Successfully processed webpage using {tool_name}. Here's the first part of the content (truncated due to length):\n\n{content[:4000]}..."
            else:
                return f"Successfully processed webpage using {tool_name}. Here's the content:\n\n{content}"
        except Exception as tool_error:
            print(f"[OldBot] Exception when invoking tool: {str(tool_error)}")
            return f"Error when processing webpage: {str(tool_error)}"
    
    
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        temperature=0.0,
        max_tokens=512,
        max_retries=2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    agent = create_react_agent(model, tools)
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": "web-thread"}}
    )
    print("[OldBot] Web agent returned a response.")
    return response["messages"][-1].content


async def get_combined_response(question, ctx=None, force_markdown=False):
    print(f"[OldBot] Processing combined response for question: {question}")
    context = await get_context(question)
    
    # Check if this might be a request for an MCP tool
    mcp_keywords = ["calculate", "compute", "sum", "add", "subtract", "multiply", "divide", 
                  "scrape", "crawl", "extract", "website", "web", "url", "http",
                  "search", "find", "query", "text matching", "analyze", "understand", 
                  "summarize", "smart extract", "intelligently"]
    
    mcp_result = None
    if any(keyword in question.lower() for keyword in mcp_keywords):
        mcp_result = await get_mcp_result(question)
    
   
    if mcp_result and len(mcp_result) > 2000 and ctx:
        # This is a large scraping result, send as file directly
        await send_response(ctx, mcp_result, force_markdown=True)
        return "I've shared the scraped content as a file for better readability."
    
    combined_prompt = f"""You are a helpful Discord bot that remembers past conversations, retrieves information from a knowledge base, and can call various MCP tools.

INFORMATION FROM KNOWLEDGE BASE:
{context}

MCP TOOL RESULT (if applicable):
{mcp_result if mcp_result else "No MCP tool was triggered."}

Based on the above and your memory, please provide a comprehensive response to the user's question.
User's Question: {question}
Answer:"""
    print("[OldBot] Combined prompt constructed.")
    messages = [{"role": "user", "content": combined_prompt}]
    response = await memory_agent.ainvoke(
        {"messages": messages},
        config={"configurable": {"thread_id": "default-thread"}}
    )
    print("[OldBot] Memory agent returned a response.")
    return response["messages"][-1].content


@bot.event
async def on_ready():
    print(f"[OldBot] {bot.user} has connected to Discord!")

@bot.command(name="power")
async def power(ctx, *, question):
    """Power command that combines knowledge lookup, web scraping, and memory"""
    try:
        
        force_markdown = "markdown" in question.lower().split()
        if force_markdown:
            question = question.replace("markdown", "", 1).strip()
        
        await ctx.send("Processing your request...")
        answer = await get_combined_response(question, ctx, force_markdown)
        
        
        await send_response(ctx, answer, force_markdown)
        
    except Exception as e:
        print(f"[OldBot] Error occurred: {e}")
        await ctx.send(f"Sorry, I was unable to process your request. Error: {str(e)}")

@bot.command(name="email")
async def email(ctx, *, query):
    """Command specifically for email-related operations"""
    try:
        await ctx.send("Processing your email request...")
        
       
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)  # 2 second timeout
            result = sock.connect_ex(('localhost', 8000))
            sock.close()
            if result != 0:
                await ctx.send("Error: Gmail MCP server is not running on port 8000. Please start the server with `python gmail_mcp.py` in the Gmail-mcp-server directory.")
                return
        except Exception as socket_error:
            print(f"[OldBot] Socket check error: {str(socket_error)}")
            
        
        
        try:
            async with MultiServerMCPClient(
                {
                    "gmail": {
                        "url": "http://localhost:8000/sse",
                        "transport": "sse",
                    }
                }
            ) as client:
                tools = client.get_tools()
                print(f"[OldBot] Available email tools: {[t.name for t in tools]}")
                
                if not tools:
                    await ctx.send("Gmail server is running but no tools are available. Check if your .env file is configured correctly with SMTP_USERNAME and SMTP_PASSWORD.")
                    return
                
                
                is_schedule_request = False
                
                
                scheduling_keywords = ["schedule", "later", "future", "remind", "send at", "send on"]
                immediate_keywords = ["right now", "immediately", "now", "right away", "instantly", "asap"]
                
                has_scheduling_words = any(keyword in query.lower() for keyword in scheduling_keywords)
                has_immediate_words = any(keyword in query.lower() for keyword in immediate_keywords)
                
                # If it has scheduling keywords but also immediate keywords, prioritize immediate sending
                if has_scheduling_words and not has_immediate_words:
                    # Check for time pattern that looks like scheduling
                    time_pattern = re.search(r'(at|on)\s+([\w\s\d,:.-]+)', query, re.IGNORECASE)
                    if time_pattern:
                        is_schedule_request = True
                
                if is_schedule_request:
                   
                    schedule_match = re.search(r'(send|schedule).*?(on|at|for)\s+([\w\s\d,:.-]+)', query, re.IGNORECASE)
                    schedule_time = None
                    
                    if schedule_match:
                        time_str = schedule_match.group(3).strip()
                        try:
                            # Attempt to parse the date-time string
                            
                            try:
                                schedule_time = dt.datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                            except ValueError:
                                # Try with more natural language - dateparser handles am/pm correctly
                                schedule_time = dateparser.parse(time_str, settings={'PREFER_DATES_FROM': 'future'})
                                
                            if schedule_time and schedule_time > dt.datetime.now():
                                
                                schedule_time_iso = schedule_time.isoformat()
                                
                                # Extract email details
                                parts = query.split("schedule", 1)
                                if len(parts) < 2:
                                    parts = query.split("later", 1)
                                if len(parts) < 2:
                                    parts = query.split("remind", 1)
                                
                                if len(parts) >= 2:
                                    email_query = parts[1].split(schedule_match.group(0))[0] + parts[1].split(schedule_match.group(0))[1]
                                else:
                                    email_query = query
                                
                               
                                recipient_match = re.search(r'to\s+([^\s]+@[^\s]+)', email_query)
                                subject_match = re.search(r'subject\s+["\'](.+?)["\']', email_query)
                                body_match = re.search(r'body\s+["\'](.+?)["\']', email_query)
                                
                                recipient = recipient_match.group(1) if recipient_match else None
                                subject = subject_match.group(1) if subject_match else "Scheduled Email"
                                body = body_match.group(1) if body_match else "This is a scheduled email."
                                
                                if not recipient:
                                    await ctx.send("I couldn't determine who to send the email to. Please include 'to recipient@example.com' in your request.")
                                    return
                                
                                
                                try:
                                    result = client.invoke_tool("gmail", "schedule_email", {
                                        "recipient": recipient,
                                        "subject": subject,
                                        "body": body,
                                        "schedule_time": schedule_time_iso
                                    })
                                    await send_response(ctx, f"Email scheduled for {schedule_time.strftime('%Y-%m-%d %H:%M')}\n\n{result}")
                                except Exception as e:
                                    await ctx.send(f"Error scheduling email: {str(e)}")
                        except Exception as parse_error:
                            await ctx.send(f"I couldn't understand the scheduling time: {str(parse_error)}\nPlease use a format like 'YYYY-MM-DD HH:MM' or a natural language time description.")
                else:
                    # Process the standard email request
                    result = await handle_email_request(query, tools)
                    await send_response(ctx, result, force_markdown=False)
        except ConnectionRefusedError:
            await ctx.send("Error: Cannot connect to Gmail MCP server. Please make sure it's running on port 8000.")
        except asyncio.TimeoutError:
            await ctx.send("Error: Connection to Gmail MCP server timed out. The server might be unresponsive.")
        except Exception as e:
            error_msg = str(e)
            print(f"[OldBot] Gmail MCP connection error: {error_msg}")
            
            if "Connection refused" in error_msg:
                await ctx.send("Error: Cannot connect to Gmail MCP server. Please make sure it's running on port 8000.")
            elif "TaskGroup" in error_msg:
                await ctx.send("Error: There was an issue with the Gmail MCP server. Please check if:\n1. The server is running on port 8000\n2. Your .env file has the correct SMTP_USERNAME and SMTP_PASSWORD\n3. You have internet access to connect to Gmail")
            else:
                await ctx.send(f"Error connecting to Gmail MCP server: {error_msg}")
    
    except Exception as e:
        await ctx.send(f"Error processing email command: {str(e)}")

if __name__ == "__main__":
    if not os.environ.get("DISCORD_TOKEN"):
        raise ValueError("DISCORD_TOKEN environment variable not set")
    bot.run(os.environ.get("DISCORD_TOKEN"))