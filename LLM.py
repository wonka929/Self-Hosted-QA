from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
import os



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
docsearch = Chroma(persist_directory="db", embedding_function=embeddings)


from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


llm=HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0, "max_length":512})

chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff", retriever=docsearch.as_retriever(),input_key="question")

#llm = HuggingFacePipeline.from_model_id(model_id="declare-lab/flan-alpaca-large", task="text-generation", model_kwargs={"temperature":0, "max_length":1000})
#qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=docsearch.as_retriever(), return_source_documents=True)
#chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever=docsearch.as_retriever())


#chain({"question": input()})
#qa({"query": input("Scrivi qui la tua ricerca:")})
try:
    response = chain.run(input('Scrivi qui la tua domanda: '))
    print(response)
except Exception as e:
    print(e)
