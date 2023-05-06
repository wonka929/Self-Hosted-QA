from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#with open(r"C:\Users\Francesco\Desktop\The Psychology of Totalitarianism 2022- Mattias Desmet.txt", encoding='utf-8') as f:
#    book = f.read()

#loader = [UnstructuredFileLoader(r"C:\Users\Francesco\Desktop\The Psychology of Totalitarianism 2022- Mattias Desmet.pdf", encoding='utf-8'), UnstructuredFileLoader(r"C:\Users\Francesco\Desktop\The unhappiness machine.pdf", encoding='utf-8')]
#raw_documents = AnalyticDB.from_documents(loader, )

loader = DirectoryLoader(r"C:\Users\Francesco\Desktop\books", silent_errors=True, show_progress=True)
raw_documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap  = 100)
documents = text_splitter.split_documents(raw_documents)

docsearch = Chroma.from_documents(
    documents, embeddings, persist_directory="db", metadatas=[{"source": f"{i}-pl"} for i in range(len(documents))]
)
docsearch.persist()
docsearch = None