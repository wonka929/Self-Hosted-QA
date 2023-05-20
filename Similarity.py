from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
docsearch = Chroma(persist_directory="db", embedding_function=embeddings)

## query the database using similarity search

while True:
	query = input("Inserisci le chiavi di ricerca: ")

	if query == "exit":
	    break
	else:
	    docs = docsearch.similarity_search_with_score(query, k=3)
	    for doc in docs:
	        print("\n")
	        print(doc[0].page_content)
	        print(doc[0].metadata["source"])
