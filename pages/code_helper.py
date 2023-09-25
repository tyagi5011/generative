from langchain.llms import OpenAI
from langchain.memory  import ConversationBufferMemory
from langchain.memory  import StreamlitChatMessageHistory
import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


os.environ['OPENAI_API_KEY'] = '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'


msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(memory_key="history", chat_memory= msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello, welcome to Coders GPT")
   
# streamlit UI

st.set_page_config(page_title='Coders GPT')
st.title('Coders GPT')
st.write("Coders GPT is traied to answer only programming related questions. It will not answer anything else")
hide_st_style = """
<style>
#MainMenu{visibility:hidden}
footer {visibility: hidden}
header {visibility: hidden}

"""
st.markdown(hide_st_style,unsafe_allow_html=True)

view_messages = st.expander("See the response")

template = """
Your are a very experienced programmer and you have to answer users query related to programming, development etc. \

if the user asked something else not related to programming then say "It is not related to programming" \

if you have to write code for the user then use the below format to delimite the code 

```programming language
   code here
```
Explaination : (if any)

Example : 
```python
   code here
```
Explaination : (if any)
{history}
Human : {human_input}
AI :
"""


prompt = PromptTemplate(template=template, input_variables=['history', 'human_input'])
chain = LLMChain(prompt=prompt,llm=OpenAI(),memory=memory)

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input(placeholder="your message"):
    st.chat_message("human").write(prompt)
    response = chain.run(prompt)
    st.chat_message("AI").write(response)




