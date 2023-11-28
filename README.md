# Local LLM Langchain ChatBot

## Introduction

The Local LLM Langchain ChatBot a tool designed to simplify the process of extracting and understanding information from archived documents. At the heart of this application is the integration of a Large Language Model (LLM), which enables it to interpret and respond to natural language queries about the contents of loaded archive files.

This application is particularly useful for those who need to sift through extensive archives and extract meaningful insights without manually reviewing each document. It's an asset for researchers, data analysts, historians, and anyone dealing with large volumes of archived material.

## Key Technologies

- **Langchain**: This is the cornerstone of the application, integrating Large Language Models (LLMs) to interpret and respond to natural language queries.
- **ChromaDB**: Used for efficient data management and retrieval, ChromaDB enhances the application's ability to handle and process large datasets.
- **FAISS** (Facebook AI Similarity Search): Employs efficient similarity search and clustering of dense vectors, crucial for processing and retrieving information from the archives.
- **HuggingFace Hub**: The application utilizes HuggingFace Hub to download and implement the necessary models, ensuring access to the latest and most efficient AI models.
- **HuggingFaceInstructEmbeddings**: This plays a pivotal role in the application by leveraging the power of Sentence Transformers for semantic search. It uses the INSTRUCTOR method, an instruction-finetuned text embedding model capable of generating text embeddings tailored to various tasks and domains. INSTRUCTOR embeds text inputs with instructions explaining the use case, enabling it to adapt to different downstream tasks and domains without additional training. This approach is key for efficient and accurate semantic search, enabling the application to find documents or text passages that are semantically related to user queries.
- **HuggingFaceInstructEmbeddings with Instructor Method**: This plays a pivotal role in the application by leveraging the power of Sentence Transformers for semantic search. It uses the INSTRUCTOR method, an instruction-finetuned text embedding model capable of generating text embeddings tailored to various tasks and domains. INSTRUCTOR embeds text inputs with instructions explaining the use case, enabling it to adapt to different downstream tasks and domains without additional training. This approach is key for efficient and accurate semantic search, enabling the application to find documents or text pass
  - **Semantic Understanding**: Unlike traditional word-level embeddings, Sentence Transformers consider the entire context of a sentence, leading to a more nuanced understanding of its meaning.
  - **Efficiency in Semantic Search**: By converting sentences into dense vector spaces, these models enable efficient similarity comparisons, crucial for semantic search applications.
- **LLAMACPP**: A Python interface for the LLaMA model, offering efficient interaction with the language model.

## Models

In this context, models refer to pre-trained artificial intelligence systems that have been developed to perform specific tasks, such as understanding natural language or generating text responses. These models are crucial for the application's ability to process and respond to user queries accurately and efficiently.

- **Interpretation and Response Generation**: The base model (`LLaMA-2-7b-chat`) interprets user queries and generates appropriate responses, facilitating an interactive and engaging user experience.
- **Semantic Search and Information Retrieval**: The embedded model (`all-MiniLM-L6-v2`) plays a vital role in semantic search, helping the application to understand and match the context of queries with the archived data.

For more information about these technologies, visit:
- [Langchain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [FAISS on GitHub](https://github.com/facebookresearch/faiss)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)
- [LLAMACPP on GitHub](https://github.com/ggerganov/llama.cpp)
- [Instructor Embedding Project](https://instructor-embedding.github.io/)
- [LLaMA-2-7b-chat Model on HuggingFace](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
- [all-MiniLM-L6-v2 on HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Modules

The Local LLM Langchain ChatBot is organized into several modules, each handling specific aspects of its functionality. This modular approach enhances the readability, maintainability, and scalability of the application. Below is an overview of each module:

- `constants.py`: Contains key variables like model names, directory paths, and other settings that remain constant throughout the application's lifecycle.
- `load_models.py`: Includes functions to load various types of models (quantized, full models) and manages device-specific configurations (CPU, GPU).
- `prompt_template.py`: Contains functions to generate and manage different prompt templates, ensuring flexibility and context relevance in user interactions.
- `qa_pipeline.py`: Initializes the QA system, incorporating elements like embeddings, Chroma vector store, and retrieval mechanisms. It orchestrates the interaction between the language models and the data retrieval process.

Each module is designed to function cohesively with others, ensuring that the application runs smoothly and efficiently. The separation into modules also makes it easier to update or extend individual components without affecting the entire system.

## How the Application Works

The application leverages a series of sophisticated technologies and models to provide an interactive question-answering service. At its core is the `retrieval_qa_pipeline` function, which sets up the necessary components for this service.

1. **Initializing the QA Pipeline**:
   - The `main` function begins by setting up the QA(Question Answering) system through `retrieval_qa_pipeline`, which involves several key steps:
     - **Embeddings Initialization**: It starts by initializing `HuggingFaceInstructEmbeddings` with the model specified in `EMBEDDING_MODEL_NAME`. This is crucial for efficient and accurate information retrieval relevant to user queries.
     - **Vector Store Setup**: If `chroma_db_store` is enabled, pre-computed embeddings from Chroma are used. Otherwise, documents are loaded from a directory, split into chunks, and converted into embeddings using FAISS.
     - **Retriever Configuration**: A retriever is set up to fetch documents relevant to queries, based on the embeddings.
     - **Language Model Loading**: A language model is loaded for generating answers, based on the retrieved information.
     - **QA System Initialization**: Depending on the `use_history` flag, a `RetrievalQA` object is initialized with appropriate parameters to combine the retriever and the language model for answering queries.

2. **Processing User Queries**:
   - The application then enters an interactive loop, where it prompts the user to enter queries.
   - For each query, the QA system processes it and retrieves the answer along with relevant source documents, if available.
   - The answers and optionally the sources are displayed to the user.
   - Users can exit the loop by entering "exit".
   - If enabled, the Q&A pairs are logged to a CSV file for record-keeping.

3. **Logging and Setup**:
   - The application configures logging to track its operations and provides feedback on the current configuration, such as the device type and whether source documents are displayed.
   - It ensures the necessary model directories are created and manages the different configurations set through command-line options.

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
## Ingesting files

The `ingest.py` script in the Local LLM Langchain ChatBot efficiently transforms archive files into a SQLite3 database for querying and analysis. It starts by loading documents from the source directory, utilizing multithreading for parallel processing. Documents are then split into chunks, and embeddings are generated for each chunk using HuggingFaceInstructEmbeddings. These embeddings are stored in Chroma, a vector store, which ultimately compiles the data into a SQLite3 database. This streamlined process ensures efficient processing and quick retrieval of data, making it ideal for handling large volumes of archived material."

### Usage
To run the `ingest.py` script, use the following command:
```bash
python ingest.py --device_type [chosen_device]
```

This command will start the ingestion process, loading and processing all documents in the specified source directory.
