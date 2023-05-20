from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

loader = DirectoryLoader(r"/home/francesco/GITHUB/Self-Hosted-QA/books/", silent_errors=True, show_progress=True)
raw_documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap  = 20)
documents = text_splitter.split_documents(raw_documents)

docsearch = Chroma.from_documents(
    documents, embeddings, persist_directory="db"
)
docsearch.persist()
docsearch = None
