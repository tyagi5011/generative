import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
import streamlit as st
from langchain.text_splitter import TextSplitter,CharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import PyPDFLoader
from pypdf import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.document_loaders import AzureBlobStorageContainerLoader



os.environ['OPENAI_API_KEY'] = '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'
os.environ['AZURE_COGNITIVE_SEARCH_SERVICE_NAME'] = 'learninglangchain'
os.environ['AZURE_COGNITIVE_SEARCH_INDEX_NAME'] = 'langchain'
os.environ['AZURE_COGNITIVE_SEARCH_API_KEY'] = 'iQ2SiK3rZF0xkC98z3GavrVsDoRJUSlHkgijLceVgLAzSeBFXg0N'



vector_store_address: str = "https://learninglangchain.search.windows.net"
vector_store_password: str = "iQ2SiK3rZF0xkC98z3GavrVsDoRJUSlHkgijLceVgLAzSeBFXg0N"
model: str = "text-embedding-ada-002"
connection_string = "DefaultEndpointsProtocol=https;AccountName=langchain2811983286;AccountKey=ML3pi2f5Mc4v8Sycr9T1ERsuNSJblLo8uqOID6pyAFbuGpe1QWlIp4Y8CUlnsQiFJbzo9lGBCTPF+ASt9aODjw==;EndpointSuffix=core.windows.net"  #


# message memory related

# msgs = StreamlitChatMessageHistory(key="langchain_messages")
# memory = ConversationBufferMemory(chat_memory=msgs,memory_key="history")

# if len(msgs.messages) == 0:
#     msgs.add_ai_message("Hello, How are you?")

embedding = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=10)
# initialize cognitive search

def indexData():
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=10)
    index_name: str = "langchain"
    loader = AzureBlobStorageContainerLoader(
        conn_str=connection_string,
        container="my-container"
    )
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=10)
    text = text_splitter.split_documents(document)
    vector_db = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=embedding.embed_query,
    )
    vector_db.add_documents(documents=text)

# chain

prompt_template = """You are a helpful assistant and provide information on the provided vector data
    {context}

    Human: {question}
    AI :"""
PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

retriever = AzureCognitiveSearchRetriever(content_key="content", top_k=10)
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.1,model="gpt-3.5-turbo"),
    retriever=retriever,
    # memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT},

)


# steamlit UI configuration

st.set_page_config(page_title="Azure Cognitive search")
st.title("Azure Cognitive search")
st.sidebar.success("Select a page above.")

hide_st_style = """
<style>
#MainMenu{visibility:hidden}
footer {visibility: hidden}
header {visibility: hidden}

"""
st.markdown(hide_st_style,unsafe_allow_html=True)


st.write("This application indexes the data from Azure blob storage to Azure cognitive \
         search. This application also facilitates user to search over that data. ")

st.button(label="Index/Refresh blob storage to Cognitive search", on_click=indexData)



   

# for msg in msgs.messages:
#     st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message('Human').write(prompt)
    response = chain.run({"question":prompt, "chat_history": []} )
    st.chat_message('AI').write(response)


    
         

        


        

       






