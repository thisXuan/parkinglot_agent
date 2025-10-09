# RAG Agent

A powerful Retrieval-Augmented Generation (RAG) agent built with LangChain, featuring multi-turn conversations, session management, and streaming responses.

## Features

- **RAG Architecture**: Combines vector search with LLM for accurate, context-aware responses
- **Multi-turn Conversations**: Persistent session management with conversation history
- **Tool Integration**: Extensible tool system using LangChain's agent framework
- **Streaming Responses**: Real-time SSE (Server-Sent Events) for better UX
- **Session Management**: Built-in session handling with Squirrel memory system
- **MCP Server Support**: Integration with Model Context Protocol servers

## Project Structure

```
agent/
├── rag_agent.py              # Core RAG agent implementation
├── nest-handler.py           # Flask API server with SSE streaming
├── llms.py                   # LLM utilities and query processing
├── requirements.txt          # Python dependencies
├── memory/
│   ├── SquirrelMemory.py    # Memory management
│   └── squirrel_session_manager.py
├── tools/
│   ├── tool.py              # Tool definitions
│   └── rag/
│       ├── embedding_oper.py # Vector store operations
│       └── perpare_data.py   # Data preparation
└── mcp-server/              # MCP server implementations
    ├── catpaw.py
    └── cursor.py
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Update API credentials in `rag_agent.py`:

```python
api_key = "your_api_key"
base_url = "your_base_url"
```

### Run the Server

```bash
python nest-handler.py
```

Server runs on `http://localhost:8000`

### API Usage

**Chat Endpoint** (Streaming):
```bash
POST /api/workstation/agent
Content-Type: application/json

{
  "query": "Your question here",
  "session_id": "optional_session_id"
}
```

**Get Session History**:
```bash
GET /api/workstation/agent/history/<session_id>
```

**Clear Session**:
```bash
POST /api/workstation/agent/clear/<session_id>
```

## Development

### Adding New Tools

1. Define your tool in `tools/tool.py`
2. Register it in the tools list
3. The agent will automatically discover and use it

### Customizing the Prompt

Modify the system prompt in `rag_agent.py` in the `_create_agent` method to adjust agent behavior.

### Vector Store Setup

Prepare your knowledge base:
```bash
python tools/rag/perpare_data.py
```

## Key Dependencies

- **LangChain**: Agent framework and LLM integration
- **Flask**: Web server and API
- **OpenAI**: LLM interface
- **torch**: Vector operations
- **rank_bm25**: Hybrid search capabilities
