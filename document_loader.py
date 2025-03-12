import os
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

class DocumentLoader:
    def __init__(self, chroma_path="chroma"):
        self.chroma_path = chroma_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def load_single_pdf(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splits = self.text_splitter.split_documents(pages)
        return splits

    def load_pdf_directory(self, directory_path):
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found at {directory_path}")

        loader = PyPDFDirectoryLoader(directory_path)
        pages = loader.load()
        splits = self.text_splitter.split_documents(pages)
        return splits

    def setup_chroma(self, documents):
        client = chromadb.PersistentClient(path=self.chroma_path)
        collection = client.get_or_create_collection("document_collection")

        # Process documents for storage
        ids = [str(i) for i in range(len(documents))]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Add documents to collection
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        return collection
