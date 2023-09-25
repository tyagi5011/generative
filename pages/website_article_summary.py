import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import streamlit as st
import time
response:  (str) = ""
os.environ['OPENAI_API_KEY']= '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'
st.set_page_config(page_title="Sentiment Detector")
st.title("Article QA")
st.write(
    "Provide link of the website and start questioning"
)
def detect_sentiment(input):
#   print(input)
    templete = """Your task is to find the sentiment from the provided text. provide the sentiment in this format only.\
         if the text is hateful or negative say : Negative Sentiment \
         if text is calm and relaxing or positive say  : Positive Sentiment
         else say : Moderate Sentiment
         {text} """
    prompt = PromptTemplate(template=templete,input_variables=['text'])
    chain = LLMChain(prompt=prompt,llm = OpenAI())
    response = chain.run(input)
    my_bar = st.progress(0, text="Detecting  Sentiment ...")
    for i in range(100):
        time.sleep(0.02)
        my_bar.progress(i+1,text="Detecting Sentiment ...")
    time.sleep(1)
    my_bar.empty()
    print(response)
    response = response.strip()  # This removes whitespace
    print(response == "Positive Sentiment")

    if (response == "Negative Sentiment") :
        st.error(icon="😡", body=response)
    elif (response == "Positive Sentiment") :
        st.success(icon="😀", body=response)
    else :
        st.info(response)

    return response


def summarise_text(input):
#   print(input)
    templete = "Your task is to summarise the provided text. {text} "
    prompt = PromptTemplate(template=templete,input_variables=['text'])
    chain = LLMChain(prompt=prompt,llm = OpenAI())
    response = chain.run(input)
    my_bar = st.progress(0, text="Detecting  Sentiment ...")
    for i in range(100):
        time.sleep(0.02)
        my_bar.progress(i+1,text="Detecting Sentiment ...")
    time.sleep(1)
    my_bar.empty()
    st.write(response)

    return response

def correct_grammer(input):
#   print(input)
    templete = "Your task is to correct the grammer of provided text. {text} "
    prompt = PromptTemplate(template=templete,input_variables=['text'])
    chain = LLMChain(prompt=prompt,llm = OpenAI())
    response = chain.run(input)
    my_bar = st.progress(0, text="Detecting  Sentiment ...")
    for i in range(100):
        time.sleep(0.02)
        my_bar.progress(i+1,text="Detecting Sentiment ...")
    time.sleep(1)
    my_bar.empty()
    st.write(response)
    return response

# option  = st.selectbox("Select an option", ['Detect Sentiment','Summarise Text','Correct Grammer'])
# website =  st.text_input("Enter website link")
# input =  st.text_input("Enter your question to search on website",disabled= not website)


if website := st.text_input("Enter website link"):
    if input := st.text_input("Enter your question to search on website",disabled= not website):
    
    # if(option == 'Detect Sentiment'):
        loader = WebBaseLoader(website)
        data = loader.load()
        embedding  = OpenAIEmbeddings()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(data)
        vector_db = Chroma.from_documents(documents=documents,embedding=embedding)
        response = vector_db.similarity_search(input)
        st.write(response[0].page_content)




    #     # detect_sentiment(input)
    # elif(option == 'Summarise Text'):
    #     summarise_text(input)
    # elif(option == 'Correct Grammer'):
    #     correct_grammer(input)

    
    
