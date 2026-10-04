# OpenCopilot: Local RAG Assistant

OpenCopilot is an offline-first, Retrieval-Augmented Generation (RAG) assistant designed for students and developers. It uses local Hugging Face embeddings, ChromaDB for vector storage, and an open-weight language model via the Groq API to provide instant, grounded answers from your project PDFs.

## Features
- **Privacy-First Ingestion:** Document chunking and embedding generation happen entirely on your local machine.
- **Open-Weight Core:** Powered by the open `openai/gpt-oss-20b` model via Groq's high-speed inference engine.
- **Interactive UI:** A clean, persistent chat interface built with Streamlit.

## Prerequisites

- Python 3.9 or higher
- A free Groq API key (get yours at [console.groq.com](https://console.groq.com/))

## Environment Setup (.env)
- You must configure your Groq API key for the LLM to generate responses. OpenCopilot uses python-dotenv to securely load this key without hardcoding it into the application.
- In the root directory of the project, create a new file named exactly .env.
- Add your Groq API key to the file in the following format (no quotation marks or spaces around the equals sign):
      GROQ_API_KEY=gsk_your_actual_api_key_characters_here
