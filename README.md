# MCP-compatible-Agentic-discord-bot
Agentic Retrieval-Augmented Generation (RAG) system built with LangChain, Chroma, and multi-MCP tools for intelligent email and web scraping integration.



# Agentic RAG with Multi-MCP Tools

Agentic Retrieval-Augmented Generation (RAG) system built with LangChain, Chroma, and multi-MCP tools for intelligent email and web scraping integration.

## Overview

This project is a sophisticated agentic RAG system that integrates a Chroma vector database with a LangChain agent to enhance the response quality by incorporating historical conversation context and real-time external tool responses. The system is deployed as a Discord bot and is capable of utilizing multiple Modular Communication Protocol (MCP) tools. Two primary MCP tools are integrated:

- **Gmail MCP Server:** Provides email functionalities such as scheduling, sending, and fetching emails.
- **Web Scraping MCP Server (using crawl4ai):** Enables dynamic web extraction and content processing.

## Features

- **Intelligent Retrieval:** Uses a Chroma vector database to split and index documents for efficient context retrieval.
- **Memory-Enhanced Interaction:** Remembers past conversation context for a more coherent dialogue.
- **Multi-MCP Integration:** Leverages specialized MCP tools to handle tasks like email operations and web scraping.
- **Discord Bot Integration:** Interacts with users through Discord commands, providing seamless access to the system’s capabilities.
- **Dynamic Tool Invocation:** Automatically routes requests based on keywords, whether for email management or web content extraction.

## Getting Started

### Prerequisites

- **Python 3.8+**
- A Discord account and server for bot deployment
- Environment variables properly set up in a `.env` file including:
  - `DISCORD_TOKEN`
  - `GOOGLE_API_KEY`
  - `MISTRAL_API_KEY`
  - `SMTP_USERNAME` and `SMTP_PASSWORD` (for Gmail MCP)

### Installation

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/yourusername/agentic-rag-mcp.git](https://github.com/MaitreyaM/MCP-compatible-Agentic-discord-bot)
   cd agentic-rag-mcp

2.	Create a virtual environment and install dependencies:
   
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt

3.	Set up your environment variables:
       Create a .env file in the project root with the following content:

        DISCORD_TOKEN=your_discord_token_here
        GOOGLE_API_KEY=your_google_api_key_here
        MISTRAL_API_KEY=your_mistral_api_key_here
        SMTP_USERNAME=your_smtp_username_here
        SMTP_PASSWORD=your_smtp_password_here

   
Running the Bot
	1.	Start the required MCP servers:
	    •	For Gmail MCP, ensure it is running on port 8000.
	    •	For the Web Scraping MCP (crawl4ai), ensure it is running on port 8001.
	2.	Run the Discord bot: 
      python power.py
  3.	Interact with the bot:
      Use the Discord commands (e.g., !power <your_query> or !email <your_query>) in your Discord server to interact with the system.

Acknowledgments
	•	Built using LangChain
	•	Vector database powered by Chroma
  •	Web scraping using craw4ai.
  •	Email functionalities using gmail smtp and imap functionalities.
	•	MCP server functionalities provided via Fastmcp.

---

This README gives a comprehensive introduction, setup instructions, and an overview of the project's capabilities, perfect for a GitHub repository.
       
