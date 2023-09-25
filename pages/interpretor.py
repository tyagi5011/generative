from langchain.llms import OpenAI
from langchain.memory import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
import streamlit as st
from langchain.memory import ConversationBufferMemory
import time

# setting OPENAI_KEY in enviourment

os.environ['OPENAI_API_KEY'] = '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'


# streamlit UI

st.set_page_config(page_title="Code interpretor,dubugger and converter")
st.title("Code interpretor,dubugger and converter")

st.write("This application do the following:")


convert_language = "none"
convert_language = st.selectbox(
    'Enter programming language to convert your code',
    ('python','java','dart','javascript')
)

llm = OpenAI(temperature=0)

message_drop = st.expander("See the result")

msgs = StreamlitChatMessageHistory(key="langchain_memory")
conversation = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("Please enter your code to continue")

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

language_schema = ResponseSchema(
    name="language",
    description="Programming Language of the provided code"
)

sumary_schema = ResponseSchema(
    name="summary",
    description="Summary of the provided code"
)

code_schema = ResponseSchema(
    name="converted_code",
    description="converted programming code to another programming language"
    
)

debug_schema = ResponseSchema(
    name="debug",
    description= "Debug the provided code and provide errors. if there ane no error say no error"
)

tip_schema = ResponseSchema(
    name="tip",
    description= "Provides a tip to improve the provided code"
)

overall_schema = [language_schema, code_schema,sumary_schema,debug_schema]
output_parser = StructuredOutputParser.from_response_schemas(overall_schema)
format_instruction = output_parser.get_format_instructions()
template = """
Your are a very experienced assistent with a great knowledge of programming and development.\
user will provide you code, your task is to find the programming language of provide code, \
summary of code, convert the code to another programmin language if asked to do so.

For the following code extract the following information

language : Detect the Programming language of the provided code and output programming language, if the provided code is not a programmin language then say "Not a programming language"
summary : Summary of the provided code
converted_code : Convert the provided code to provided programming language : {convert_language}

debug : Check the code for errors. check for each error like syntax error or any other error then output the error if no error than output no error
the input code is delimited by triple backticks \
converted_code : convert to any other programming language

tip : Provide a tip to improve the provided code


code: {human_input}


{format_instruction}


"""

prompt = PromptTemplate(input_variables=[ "human_input", 'format_instruction','convert_language'], template=template)
# formatted_prompt = prompt.format_prompt(human_input=input, format_instruction=format_instruction)
chain = LLMChain(llm=llm,prompt= prompt,verbose=True)
global_response = {}


if input := st.chat_input():
    st.chat_message("human").write(input)
    formatted_prompt = prompt.format_prompt(human_input=input, format_instruction=format_instruction, convert_language=convert_language)
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()    
    response = chain.run({'human_input':input, 'format_instruction':format_instruction, 'convert_language': convert_language})
    print(response)
    try:
        json = output_parser.parse(response)
        global_response = json
        print(json)
        with st.chat_message("AI").container():
            # with message_drop:
            #     message_drop.json(global_response)

            st.info(f"Programmin language of code is {json['language']}")
            st.text("summary of code:")
            st.info(json['summary'])
            st.text('Errors in code are')
            st.error(json['debug'])
            st.text('Tip')
            st.success(json['tip'])
        
    
        # st.markdown(body=
        #     f'''<div style=" border: 1px solid #ccc; padding: 10px; margin: 10px 0;"> \
        #     <p style="margin: 0 0 10px 0;"><b>Language :</b> - {json['language']}</p>\
        #     <p style="margin: 0 0 10px 0;"><b>Summary of code:-</b> {json['summary']}</p>\
        #     <p style="margin: 0 0 10px 0;"><b>Errors:-</b> {json['debug']}</p>\
        #     </div>  \
        #     ''',
        #     unsafe_allow_html=True
            # )
            st.text(f"Converted code to {convert_language} ")
            st.code(json['converted_code'])
            st.text('Response Json' )
            st.json(json)
    
    except:
        st.text(response)

 
    
    
        
        
        