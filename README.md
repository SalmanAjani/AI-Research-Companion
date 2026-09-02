# AI Research Companion

This is an advanced Retrieval-Augmented Generation (RAG) companion that allows users to ask questions about the contents of
documents and receive answers based on the information found in them.

## What problem does this solve

Large Language Models do not automatically know the contents of private PDFs or documents.

This project solves that problem by:

1. Loading the PDFs.
2. Splitting them into smaller chunks.
3. Creating embeddings for those chunks.
4. Storing them in a vector database.
5. Retrieving relevant information when a user asks a question.
6. Compressing the retrieved information to remove unnecessary content.
7. Sending the relevant context to an LLM to generate the final answer.

It also maintains conversation history so that follow-up questions can be understood in context.

## Features

- PDF document ingestion
- Text chunking
- Semantic/vector search
- Chroma vector database
- Multi-query retrieval for better document recall
- Contextual compression of retrieved documents
- Conversation history
- Persistent vector database
- OpenRouter LLM integration
- Runnable-based LangChain pipeline
- Source/page information in retrieved context
- Interactive question-answering through the terminal

## Architecture

```text
                PDFs/CSVs/Docs
                     |
                     v
                 PDF Loader
                     |
                     v
                 Text Chunks
                     |
                     v
                 Embeddings
                     |
                     v
                Chroma Vector DB
                     |
                     |
User Question ---> Retrieval
                     |
                     v
                 Multi-Query
                     |
                     v
            Contextual Compression
                     |
                     v
              Relevant Context
                     |
                     +------ Conversation History
                     |
                     v
                    LLM
                     |
                     v
                   Answer

```

## Tech Stack

- Python - Application language
- LangChain - RAG pipeline and runnable architecture
- Chroma - Vector database
- Hugging Face Embeddings - Local text embeddings
- OpenRouter - LLM provider
- PyPDF - PDF loading

## Setup

- Install dependencies

```
    pip install -r requirements.txt
```

- Add your OpenRouter API key

```
    OPENROUTER_API_KEY=your_api_key_here
```

- Add your PDFs

```
    data/research.pdf
```

- Run the application

```
    python rag_assistant.py
```

## Features to work on to improve this further

- Use stronger embedding models for better retrieval quality.
- Add a dedicated reranker.
- Implement hybrid search using both keyword and semantic search.
- Improve chunking for documents containing tables, headings, and complex layouts.
- Provide more precise source citations and page references.
- Improve conversation memory using summarization or windowed memory.
- Add RAG evaluation to measure retrieval and answer quality.
- Add caching to reduce repeated LLM calls.
- Optimize Multi-Query Retrieval and compression to reduce latency and API costs.
- Add support for multiple document formats in the future.
