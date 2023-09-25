# importing libraries pandas, streamlit,langchain
import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import openai
import os

# creating app using stramlit and langchain and openAi


# page title
st.title("Analyze your csv documents")
os.environ["OPENAI_API_KEY"] = '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'

# hidding made with streamlit on footer

hide_st_style = """
<style>
#MainMenu{visibility:hidden}
footer {visibility: hidden}
header {visibility: hidden}

"""
st.markdown(hide_st_style,unsafe_allow_html=True)

# this is a function to upload csv files

def upload_csv(input_csv):
    df = pd.read_csv(input_csv)
    with st.expander('see dataframe'):
        st.write(df)
    return df


def generate_response(csv_file, input_query):
    llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo-0613")
    df= upload_csv(csv_file)
    agent = create_pandas_dataframe_agent(llm, df, AgentType.OPENAI_FUNCTIONS)
    response = agent.run(input_query)
    return st.success(response)

# input widgets are

uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
question_list = [
  'How many rows are there?',
  'Summerize the data from the dataframe',
  'What is this document all about',
  'Other']

query_text = st.selectbox('Select an query text',question_list,disabled=not uploaded_file)
# openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))

if query_text == 'Other' :
    query_text = st.text_input('Enter your query:',placeholder='Enter query', disabled=not uploaded_file)
# if not openai_api_key.startswith('sk-'):
#   st.warning('Please enter your OpenAI API key, If you want to use yours', icon='⚠')
if uploaded_file is not None:
  st.header('Output')
  generate_response(uploaded_file, query_text)
