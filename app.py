import os
from io import BytesIO
from PIL import Image
import google.generativeai as genai
import google.ai.generativelanguage as glm
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st

st.title("DocsGPT")

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

rag = glm.Tool(
    function_declarations=[
      glm.FunctionDeclaration(
        name='vector_search',
        description="Returns the content of the document user attached. Make sure that your not passing query as a question use like **keywords** instead. Use this function to search for contents in the user attached or uploaded documents to you. Try not to completly paste the user question as query, instead use keywords.",
        parameters=glm.Schema(
            type=glm.Type.OBJECT,
            properties={
                'query': glm.Schema(type=glm.Type.STRING),
            },
            required=['query']
        )
      )
    ]
)

gemini = genai.GenerativeModel('gemini-pro', tools=[rag])

file_path = './getting_real_basecamp.pdf'

def loader_data(file_path):
    pdf_reader = PdfReader(file_path)
    content = ''
    for page in pdf_reader.pages:
        content += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    texts = text_splitter.split_text(content)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector_store = Chroma.from_texts(texts, embeddings).as_retriever()
        st.session_state.knowdge_base = vector_store
    except:
        return None

if "history" not in st.session_state:
    st.session_state.history = []

if "knowdge_base" not in st.session_state:
    st.session_state.knowdge_base = None

if "chat" not in st.session_state:
    st.session_state.chat = gemini.start_chat(history=[glm.Content(
            parts=[glm.Part(
                text="Your name is DocsGPT. You are very helpful and can assist with documents uploaded by the user. Use the vector_search tool/function to search for contents in the user attached or uploaded documents to you.\nInitially 'getting_real_basecamp.pdf' book has been atttached with you to answer to question from user. Its all about 'The smarter, faster, easier way to build asuccessful web application'. The user'll ask questions based on the book."
            )],
            role="user"
        ),
        glm.Content(
            parts=[glm.Part(
                text="Sure, i can do that for you."
            )],
            role="model"
        )])

for history in st.session_state.history:
    with st.chat_message(history["role"]):
        st.markdown(history["text"])

if st.session_state.knowdge_base is None:
    loader_data(file_path)

if prompt := st.chat_input("Enter your message..."):
    st.session_state.history.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            response = st.session_state.chat.send_message(prompt)
            if response.candidates[0].content.parts[0].text == '':
                print('calling shit out')
                args = response.candidates[0].content.parts[0].function_call.args['query']
                print("searching for ", args)
                related_docs = str(st.session_state.knowdge_base.get_relevant_documents(args))
                print(related_docs)
                response = st.session_state.chat.send_message(
                    glm.Content(
                        parts=[glm.Part(
                            function_response = glm.FunctionResponse(
                            name='vector_search',
                            response={'rag': related_docs},
                            )
                        )]
                    )
                ).candidates[0].content.parts[0].text
            else:
                response = response.candidates[0].content.parts[0].text
            print(st.session_state.chat.history)
        except:
            response = "I'm sorry, I cannot answer that question. please try again with a different question."

        message_placeholder.markdown(response)
    st.session_state.history.append({"role": "assistant", "text": response})
