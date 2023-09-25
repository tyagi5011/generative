import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os
from pypdf import PdfReader




os.environ['OPENAI_API_KEY'] = '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'
llm = ChatOpenAI(temperature=0.1,model='gpt-3.5-turbo')
st_markdown_code  =  """
<style>
footer {visibility: hidden}
"""


def generate_response(uploaded_file, query):


    if uploaded_file is not None:
        # Check the file extension to determine the file type
        file_type = uploaded_file.type
        st.text(file_type)
    
    if file_type == "text/plain":
        documents = [uploaded_file.read().decode()]
        qa = text_embedding(documents, query)
        st.text(qa)
        return qa.run(query)

    else:
       
        document = PdfReader(uploaded_file)
        text = ''
        for pages in document.pages:
            text = text + pages.extract_text()
        text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
        texts = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_texts(texts,embeddings)
        retrieve = db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',retriever = retrieve)
        return qa.run(query)

def text_embedding(documents, query):
    # documents = "this is a sample text file"
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)
    # Select embeddings
    st.text(texts)
    embeddings = OpenAIEmbeddings()
    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)
    # Create retriever interface
    retriever = db.as_retriever()
    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=retriever)
    return qa

# streamlit elements

st.set_page_config(page_title="Chat with your data", )
st.title("Chat with your documents")

uploaded_file = st.file_uploader("Upload a file", type=['pdf'])

query = st.text_input("Enter a question", )
st.markdown(st_markdown_code,unsafe_allow_html=True)



# submitting form

with st.form('myform',clear_on_submit=True):
    submitted = st.form_submit_button('Submit' , disabled=not(uploaded_file,query))
    if submitted:
        response = generate_response(uploaded_file, query)
        st.write('Loading...')
        st.write(response)
