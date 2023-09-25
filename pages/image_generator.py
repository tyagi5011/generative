import os
from langchain.llms import OpenAI
from langchain.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import streamlit as st
import time


os.environ['OPENAI_API_KEY'] = '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'
# image_url =  ''
def generate_image(input):

    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["image_desc"],
        template="Generate an detailed image based on following description: {image_desc}",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    image_url = DallEAPIWrapper().run(chain.run(input))
    st.image(image_url)

st.set_page_config(page_title="Image Generator")
st.title("Image Generator")

if desc := st.text_input("Enter the prompt to generate an image"):
    my_bar = st.progress(0, text="generating image ...")
    for i in range(100):
        time.sleep(0.15)
        my_bar.progress(i+1,text="generating image ...")
    time.sleep(2)
    my_bar.empty()

    generate_image(desc)

   

    







# st.image(image)