import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import initialize_agent,load_tools,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os

os.environ['OPENAI_API_KEY'] = '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'

llm = ChatOpenAI(temperature=0.2, model='gpt-3.5-turbo',streaming=True)


tools = load_tools(['llm-math', 'wikipedia', 'llm-math'],llm=llm)
agents = initialize_agent(
    llm=llm,
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# stramlit ui

st.set_page_config(page_title="Advance Chatbot")
st.title("Advance Chatbot")
st.write("This chatbot has capablity of answering the following:")
st.markdown("""
1. Math Related Questions
2. Wikipedia Related Questions
3. Search Related Questions using Duck Duck Go
""")
    


if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st.write("🧠 thinking...")
        st_callback = StreamlitCallbackHandler(st.container())
        response = agents.run(prompt, callbacks=[st_callback])
        st.write(response)