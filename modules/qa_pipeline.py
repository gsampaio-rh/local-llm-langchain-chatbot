import logging

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from modules.prompt_template import get_prompt_template

from modules.load_models import (
    load_model,
)

from modules.constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    CHROMA_SETTINGS,
    MODEL_ID,
    MODEL_BASENAME
)

def retrieval_qa_pipeline(device_type, chroma_db_store, use_history, promptTemplate_type="llama"):
  """
  Initializes and returns a retrieval-based Question Answering (QA) pipeline.

  This function sets up a QA system that retrieves relevant information using embeddings
  from the HuggingFace library. It then answers questions based on the retrieved information.

  Parameters:
  - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
  - use_history (bool): Flag to determine whether to use chat history or not.

  Returns:
  - RetrievalQA: An initialized retrieval-based QA system.

  Notes:
  - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
  - The Chroma class is used to load a vector store containing pre-computed embeddings.
  - The retriever fetches relevant documents or data based on a query.
  - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
  - The model is loaded onto the specified device using its ID and basename.
  - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
  """

  embeddings = HuggingFaceInstructEmbeddings(
      model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})
  # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
  # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

  if chroma_db_store:
      # load the vectorstore
      db = Chroma(
          persist_directory=PERSIST_DIRECTORY,
          embedding_function=embeddings,
          client_settings=CHROMA_SETTINGS
      )
      retriever = db.as_retriever()
  else:
      loader = DirectoryLoader('data/',
                                glob="*.pdf",
                                loader_cls=PyPDFLoader)

      documents = loader.load()
      # print(documents)

      # ***Step 2: Split Text into Chunks***
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=500,
          chunk_overlap=50)

      text_chunks = text_splitter.split_documents(documents)
      print(len(text_chunks))
      # Convert the Text Chunks into Embeddings and Create a FAISS Vector Store***
      vector_store = FAISS.from_documents(text_chunks, embeddings)
      retriever = vector_store.as_retriever(search_kwargs={'k': 2})

  # get the prompt template and memory if set by the user.
  prompt, memory = get_prompt_template(
      promptTemplate_type=promptTemplate_type, history=use_history)

  # load the llm pipeline
  llm = load_model(device_type, model_id=MODEL_ID,
                    model_basename=MODEL_BASENAME, LOGGING=logging)

  if use_history:
      qa = RetrievalQA.from_chain_type(
          llm=llm,
          chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
          retriever=retriever,
          return_source_documents=True,  # verbose=True,
          callbacks=callback_manager,
          chain_type_kwargs={"prompt": prompt, "memory": memory},
      )
  else:
      qa = RetrievalQA.from_chain_type(
          llm=llm,
          chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
          retriever=retriever,
          return_source_documents=True,  # verbose=True,
          callbacks=callback_manager,
          chain_type_kwargs={
              "prompt": prompt,
          },
      )

  return qa
