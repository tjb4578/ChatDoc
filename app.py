# Imports
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAIChat
from langchain.chains.question_answering import load_qa_chain
import pickle
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()

    st.title("Chat with your docs using LLMs")

    # Add file uploader here
    file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if file is not None:
        # Get file extension
        store_name = os.path.splitext(file.name)[0]
        file_extension = os.path.splitext(file.name)[1]
        text = ""
        
        if file_extension == '.txt':
            # Read file
            text = file.read().decode('utf-8')

        elif file_extension == '.pdf':
            # Read file using pdf loader
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

        # Create text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        # Split text into chunks
        chunks = splitter.split_text(text)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"./pickles/{store_name}.pkl", "rb") as f:
                db = pickle.load(f)
            st.write('Embeddings Loaded from the Disk')
        else:
            embeddings = OpenAIEmbeddings()
            db = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"./pickles/{store_name}.pkl", "wb") as f:
                pickle.dump(db, f)

        # Take a query and do a semantic search over the vector store
        query = st.text_input("Ask a question")
        if query:
            docs = db.similarity_search(query, k=3)
        
            # Load the QA chain
            llm = OpenAIChat(model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)



if __name__ == "__main__":
    main()