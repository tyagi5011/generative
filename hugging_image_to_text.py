from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
import os
import requests
import streamlit as st





os.environ['OPENAI_API_KEY'] = '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'
API_TOKEN = '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'

def main():
    st.set_page_config(page_title="Image to story and audio")
    st.title("Image to story and audio")
    upload_file = st.file_uploader(label="Select image",type=['jpg'])
    if upload_file is not None:
        byte_data = upload_file.getvalue()
        with open(upload_file.name,"wb") as file:
            file.write(byte_data)
        st.image(upload_file,use_column_width=True)
        scenario = img2text('academic.jpg')
        story = generate_story(scenario)
        text_to_speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        
        st.audio("audio.flac")



def img2text (url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0] ['generated_text']
    print (text)
    return text


def generate_story(scenario):
    template = """You are a story writer. you can create a short story based on given scenario. the story should not be more than 20 words
    context : {input}
    Story:
    """

    prompt = PromptTemplate(template= template, input_variables={'input'})

    chain = LLMChain(llm=ChatOpenAI(temperature=0.8,model='gpt-3.5-turbo'), prompt=prompt,verbose=True)
    story = chain.predict(input=scenario)
    return story

def text_to_speech(text):
    import requests

    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response
        
    output = query({
        "inputs": text,
    })

    with open('audio.flac','wb') as file:
        file.write(output.content)


