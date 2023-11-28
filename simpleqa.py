import sys
import logging
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download

from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS
)

def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        elif ".awq" in model_basename.lower():
            model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.2,
        # top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm

llm = load_model("cpu", model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

#**Step 1: Load the PDF File from Data Path****
loader=DirectoryLoader('data/',
                       glob="*.pdf",
                       loader_cls=PyPDFLoader)

documents=loader.load()
#print(documents)

#***Step 2: Split Text into Chunks***
text_splitter=RecursiveCharacterTextSplitter(
                                             chunk_size=500,
                                             chunk_overlap=50)

text_chunks=text_splitter.split_documents(documents)
print(len(text_chunks))

#**Step 3: Load the Embedding Model***
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})

#**Step 4: Convert the Text Chunks into Embeddings and Create a FAISS Vector Store***
vector_store=FAISS.from_documents(text_chunks, embeddings)

##**Step 5: Find the Top 3 Answers for the Query***
query="YOLOv7 outperforms which models"
docs = vector_store.similarity_search(query)

#print(docs)


template="""Use the following pieces of information to answer the user's question.
If you dont know the answer just say you know, don't try to make up an answer.

Context:{context}
Question:{question}

Only return the helpful answer below and nothing else
Helpful answer
"""

qa_prompt=PromptTemplate(template=template, input_variables=['context', 'question'])

#start=timeit.default_timer()

chain = RetrievalQA.from_chain_type(llm=llm,
                                   chain_type='stuff',
                                   retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                   return_source_documents=True,
                                   chain_type_kwargs={'prompt': qa_prompt})

#response=chain({'query': "YOLOv7 is trained on which dataset"})

#end=timeit.default_timer()
#print(f"Here is the complete Response: {response}")

#print(f"Here is the final answer: {response['result']}")

#print(f"Time to generate response: {end-start}")

while True:
    user_input=input(f"prompt:")
    if query=='exit':
        print('Exiting')
        sys.exit()
    if query=='':
        continue
    result=chain({'query':user_input})
    print(f"Answer:{result['result']}")