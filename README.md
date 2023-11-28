# Local LLM Langchain ChatBot

## Introduction
The Local LLM Langchain ChatBotis a tool designed to simplify the process of extracting and understanding information from archived documents. At the heart of this application is the integration of a Large Language Model (LLM), which enables it to interpret and respond to natural language queries about the contents of loaded archive files.

This application is particularly useful for those who need to sift through extensive archives and extract meaningful insights without manually reviewing each document. It's an asset for researchers, data analysts, historians, and anyone dealing with large volumes of archived material.

## Key Technologies:

- **Langchain**: This forms the core of the application, integrating Large Language Models (LLMs) to interpret and respond to natural language queries.
- **ChromaDB**: Employed for efficient data management and retrieval, ChromaDB enhances the application's ability to handle large datasets.
- **FAISS** (Facebook AI Similarity Search): Used for efficient similarity search and clustering of dense vectors, crucial for processing and retrieving information from the archives.
- **HuggingFace Hub**: The application utilizes HuggingFace Hub to download and implement the necessary models, ensuring access to the latest and most efficient AI models.
- **Instructor**: Instruction-finetuned text embedding model that can generate text embeddings tailored to any task (e.g., classification, retrieval, clustering, text evaluation, etc.) and domains (e.g., science, finance, etc.) by simply providing the task instruction, without any finetuning.
- **LLAMACPP**: A Python interface for the LLaMA model, offering efficient interaction with the language model.

## Key Technologies:

- **Langchain**: This is the cornerstone of the application, integrating Large Language Models (LLMs) to interpret and respond to natural language queries.
- **ChromaDB**: Used for efficient data management and retrieval, ChromaDB enhances the application's ability to handle and process large datasets.
- **FAISS** (Facebook AI Similarity Search): Employs efficient similarity search and clustering of dense vectors, crucial for processing and retrieving information from the archives.
- **HuggingFace Hub**: The application utilizes HuggingFace Hub to download and implement the necessary models, ensuring access to the latest and most efficient AI models.
- **InstructorEmbeddings**: An instruction-finetuned text embedding model capable of generating text embeddings tailored to various tasks and domains, simply by providing the task instruction.
- **LLAMACPP**: A Python interface for the LLaMA model, offering efficient interaction with the language model.

For more information about these technologies, visit:
- [Langchain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [FAISS on GitHub](https://github.com/facebookresearch/faiss)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)
- [LLAMACPP on GitHub](https://github.com/ggerganov/llama.cpp)
- [Instructor Embedding Project](https://instructor-embedding.github.io/)

## Models:

### Base Model
The `LLaMA-2-7b-chat Model` serves as the base model. It is specifically tuned for chat-like interactions, making it ideal for interpreting and responding to user queries in a conversational manner.
- [LLaMA-2-7b-chat Model on HuggingFace](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)

### Embedded Model
The application uses the `all-MiniLM-L6-v2` model for efficient and effective text embedding. This model is known for its compact size and powerful performance in generating meaningful text representations.
- [all-MiniLM-L6-v2 on HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## How to Run the Application

Running the application is straightforward. Follow these steps to get started:

1. **Create and Activate a Virtual Environment**:
```bash
  python -m venv .env
  source .env/bin/activate
```

2. **Install Required Packages**:
```bash
  pip install -r requirements.txt
```

3. **Run the Application**:
```bash
  python localllm.py
```

### Options for Configuration
The application allows for various configurations to tailor its operation to your needs. The available options include:

- `--device_type` [cpu|cuda|ipu|xpu|...|hpu|mtia]:  Specify the device to run on. Default is `cuda` if available, otherwise `cpu`.
- `--show_sources, -s`: Enable this option to show sources along with answers. Default is False.
- `--use_history, -h`: Use this option to maintain a history of interactions. Default is False.
- `--model_type` [llama|mistral|non_llama]: Choose the model type: `llama`, `mistral`, or `non_llama`. Default is `llama`.
- `--chroma_db_store`: Enable this to use ChromaDB. Default is False.
- `--save_qa`: Set this option to save Q&A pairs to a CSV file. Default is False.

These options allow you to customize the application's performance and output according to your requirements.
