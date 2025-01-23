import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

## langchain tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] ="true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OLLAMA"

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant. PLease response to the user queries"),
        ("user","Question:{question}")
    ]
)
def generate_response(question,engine,temperature, max_tokens):
    llm = Ollama(model = engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question':question})
    return answer

# Title
st.title("Q&A chatbot")

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter API key", type="password")

# Dropdown to select various Open AI models
llm = st.sidebar.selectbox("Select an model",["gpt-4o"])

# Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("ask any question")
user_input = st.text_input("you:")

if user_input:
    response = generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")
    
## Fastapi integration
# app = FastAPI(title="Langchain server",
#               version="1.0",
#               description="A simple API server using Langchain runnable interface")

# # Adding chain routes
# add_routes(
#     app,
#     chain,
#     path="/chain"
# )

# if __name__=="__main__":
#     import uvicorn
#     uvicorn.run(app,host="localhost", port =8000)