from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

chroma_directory = 'db'

db = Chroma(persist_directory=chroma_directory, embedding_function=embeddings)

loader = DirectoryLoader(r"/home/wonka/GITHUB/Self-Hosted-QA/added/", silent_errors=True, show_progress=True)
raw_documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap  = 20)
document = text_splitter.split_documents(raw_documents)

db.add_documents(documents=document)

db.persist()

#db.similarity_search_with_score(query="La città vissuta")
# --> results from both documents
