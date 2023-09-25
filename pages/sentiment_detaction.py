import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import time
response:  (str) = ""
os.environ['OPENAI_API_KEY']= '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'
st.set_page_config(page_title="Sentiment Detector")
st.title("Text Processor")
st.write(
    "This application Detects Sentiments, Summarize text, Correct grammer from a give text"
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

option  = st.selectbox("Select an option", ['Detect Sentiment','Summarise Text','Correct Grammer'])
input =  st.text_input("Enter your text to detect the sentiment")


if input is not None:
    if(option == 'Detect Sentiment'):
        detect_sentiment(input)
    elif(option == 'Summarise Text'):
        summarise_text(input)
    elif(option == 'Correct Grammer'):
        correct_grammer(input)

    
    
