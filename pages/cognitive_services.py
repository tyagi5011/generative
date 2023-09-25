import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import AzureCognitiveServicesToolkit
from langchain.agents import load_tools, initialize_agent, AgentType

import streamlit as st
import time
response:  (str) = ""
os.environ['OPENAI_API_KEY']= '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'
os.environ["AZURE_COGS_KEY"] = "8b79787c525f45afbfe2ba1fb492cf5a"
os.environ["AZURE_COGS_ENDPOINT"] = "https://cognitivesearchlangchain.cognitiveservices.azure.com/"
os.environ["AZURE_COGS_REGION"] = "eastus"
st.set_page_config(page_title="Cognitive Services")
st.title("Azure Cognitive Services toolkit")
st.write(
    "Provide link of the website and start questioning"
)


llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
def detect_images():
#   print(input)


    # image =  st.text_input("Enter image url")
    # input =  st.text_input("Enter your question to query on/with image",disabled= not image)
    if image := st.text_input("Enter image link"):
        if input := st.text_input("Enter your query on image"):
    
            templete = """Analyze the given image and answer the query 
                image : {image}
                query : {query}
                """
            prompt = PromptTemplate(template=templete,input_variables=['image','query'])
            chain = LLMChain(prompt=prompt,llm = OpenAI())
            toolkit = AzureCognitiveServicesToolkit()

            agent = initialize_agent(
                tools=toolkit.get_tools(),
                llm = llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose = True
            )

            message = prompt.format(image=image, query= input)

            response = agent.run(image + input)
            my_bar = st.progress(0, text="Processing  Images ...")
            for i in range(100):
                time.sleep(0.02)
                my_bar.progress(i+1,text="Processing Images ...")
            time.sleep(1)
            my_bar.empty()
            print(response)
            response = response.strip()  # This removes whitespace
            st.write(response)

            return response


def text_to_speech():
    print("entered")
    st.subheader('Please Enter prompt related to speech generation. Please be sure to write speech generation related prompt ')
    st.text(
        "Example 1 : Tell me a joke, and read it out "
         
    )
    st.text("Example 2 : i am going to play with my friends, convert it to speech")
    if input := st.text_input("Enter your prompt"):
        toolkit = AzureCognitiveServicesToolkit()
        agent = initialize_agent(
            tools=toolkit.get_tools(),
            llm = llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose = True
        )
        # message = prompt.format(image=image, query= input)
        response = agent.run(input+ " output audio file only no other text. For Example : /var/folder/../audio.wav ")
        # my_bar = st.progress(0, text="Processing   ...")
        # for i in range(100):
        #     time.sleep(0.02)
        #     my_bar.progress(i+1,text="Processing  ...")
        # time.sleep(1)
        # my_bar.empty()
        # print(response)
        # response = response.strip()  # This removes whitespace
        st.write(response)
        # st.audio(data=response.strip())
        return response

def correct_grammer(input):
#   print(input)
    templete = "Your task is to correct the grammer of provided text. {text} "
    prompt = PromptTemplate(template=templete,input_variables=['text'])
    chain = LLMChain(prompt=prompt,llm = OpenAI())
    response = chain.run(input )
    my_bar = st.progress(0, text="Detecting  Sentiment ...")
    for i in range(100):
        time.sleep(0.02)
        my_bar.progress(i+1,text="Detecting Sentiment ...")
    time.sleep(1)
    my_bar.empty()
    st.write(response)
    return response

option  = st.selectbox("Select an option", ['Analyze Documents / images','Text to speech','Correct Grammer'])
# website =  st.text_input("Enter website link")
# input =  st.text_input("Enter your question to search on website",disabled= not website)


# if website := st.text_input("Enter website link"):
#     if input := st.text_input("Enter your question to search on website",disabled= not website):
    
#     # if(option == 'Detect Sentiment'):
#         loader = WebBaseLoader(website)
#         data = loader.load()
#         embedding  = OpenAIEmbeddings()
#         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#         documents = text_splitter.split_documents(data)
#         vector_db = Chroma.from_documents(documents=documents,embedding=embedding)
#         response = vector_db.similarity_search(input)
#         st.write(response[0].page_content)




if option is not None:
    if(option == 'Analyze Documents / images'):
        detect_images()
    elif(option == 'Text to speech'):
        text_to_speech()
    

    
    
