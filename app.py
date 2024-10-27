import os
import streamlit as st
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

with st.sidebar:
    st.title("LLM Research Paper Chat")
    st.markdown('''
        ## About
 Introducing the Ultimate LLM-Powered Chatbot: Your Research Assistant Redefined! 🚀
Are you ready to revolutionize the way you interact with research papers? Say hello to the chatbot, seamlessly powered by LangChain and the game-changing Gemini-AI API! This isn't just another chatbot; it’s your personal research assistant, designed to supercharge your understanding and engagement with academic literature like never before.
    ''')
    
    # Add vertical space
    for _ in range(5):
        st.write("")

    st.write("Made by Kishorekumar Vishwanathan")

def main():
    st.header("Chat with Research Paper 🗨️")
    load_dotenv()

    # Upload PDF file.
    pdf = st.file_uploader("Upload your pdf", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        #Embeddings
        store_name = pdf.name[:-4]  
         
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write("Embedddings loaded from the disk")
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        query = st.text_input("Ask questions about your Research Paper: ")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff" )
            with get_openai_callback() as cb:

                response = chain.run(input_documents=docs,question=query)
                print(cb)
            st.write(response)





if __name__ == '__main__':
    main()
