import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables and Google API credentials
load_dotenv()
os.getenv("GOOGLE_API_KEY")


import json
from google.oauth2 import service_account

credentials_path = "stellar-depth-419411-f072fec7d927.json"  # Make sure this path is correct

try:
    with open(credentials_path, "r") as file:
        credentials_info = json.load(file)
except FileNotFoundError:
    print("Error: JSON file not found. Check the file path.")
except json.JSONDecodeError:
    print("Error: Failed to decode JSON. Check the file contents.")

# Initialize the credentials if JSON was successfully loaded
if 'credentials_info' in locals():
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    print("Credentials successfully loaded.")
else:
    print("Failed to load credentials.")


import json
from google.oauth2 import service_account
import streamlit as st
# Retrieve the JSON from Streamlit secrets
credentials_info = {
  "type": "service_account",
  "project_id": "stellar-depth-419411",
  "private_key_id": "f072fec7d927683055cb4851c50a00b911cdf6f3",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCaE0Y0+kp2jk4X\n1JET7qivLsRNlum/Sr1OJOwb3UoIJBOcotII/73dx5JKndBIbY3a+ng5Os3b0ROA\nEvpXHX58a/yre2dLOlEU0sAXGY8NSvYo1Cd/NcSAk11x/nciD84IXFyMOpRG6d/H\nUUzR5BUtJ9zPmVpfgKr1MNZx2mlbWOJiXN+ALG2oRNA4IxhRWY8sleJUhgo0tiHw\nFFdoaoEXRy5d//1enMndJ29sfg7pCDqrV+KvWCqXNmCCcV8SCjeExot8OYqgyQmX\n7sv2UuFtAsntscZs6zSULVLe6abl3ki+K15wa5tKSSVzxYZGY7Veq3W68P7pf4re\ncLg/DdoPAgMBAAECggEAAOkhQOMM+XpFauR/Y06zT4nCubJkAkum3YC7pobMGErw\nAnOkMypqqF9rh7ioo9qqfDERgvyDD/nCc6zWwT19LGYXSJWd7L/5j7XgCTUz0pvK\nda0tDSilaXhHy8F5x4Ad/EbO58xraPC/XciUu5W2mgsUxfWiJwo2AZxH7emhEaUR\nti/hXqv7EoBtjhEg610+vpmIUU2x/kWQm9cCYUZITJBXLsVJmEBdt9sAtXbq9dER\nsExPGZKRyL3BpWO2wWqYv8fs3OGQ70db/1j4Pj8EHqyfvO76y6zBCriR8ACW8/l5\nOm4j+RK0T8uDqrr252WX9yDrksoDRTBeE+tAPmvydQKBgQDPVHUR5D8A+NPhQoCx\nEY4V21u2yEqL++B/uchi5RL25EiZklgifdMX0KoCp4zDDw/0R3eVMUmjKNzmdyDm\nzxsc2NHpwIFr3ZWh4vPpqaMqVNXhHUGYKM6+N5kL0VdkTIJbtvu6/x+6OEKyxIL4\n07eI6vC5eoXtbFnYFgr+z81xMwKBgQC+Pneacz1rbB/v6DNuJWRl1EccsRePzffX\nwoQUbQAcvCGKMeyhEmFjuokK1neBp0A2whoCUuSEBAOWpjJNsIzmZpawx0jVvgy/\n2JhmGT+8ZbdILiP62cUI3AJyY0e0IpbY1xM8rMDuwr01fDC22FtOSzUmR8XOjv6f\nwhco6pHrtQKBgHMtwDvISRgJI+woPcYgsoaB7lmEu6U4sGdEloYaLIbsG0j1e/Dt\nZa/9Q/Vlj1VtsLdMXKqNTxNNSCrgU27l73H/Id5yC3QZDV957XcJvpNtvcPptN8L\nDI+v414lVh9qQaEh7obb5IxXZPZbJUeGlpeBrWndHznez6qz1DfqyX7xAoGAK6K4\nXDzCgbkzOhvQcBszhAfEp9gWx20+w8Zh9S1rMSwVpVT+KZPFstI+TLYUgzCRkf3D\naXUJ5R3mlM9aCmfMaaxuM+4BzsTgt8A+dGymKdhKycuLhSYeA4IzLXmIINEuOF5c\nkzYsqpcQPwxVQBswFi0566XawR4bWRlzpnbnMWECgYEAlmlojnbERIS72QNOv6cJ\nfdqh6xgZk5LQwzT5vidHfUUzi2kZM77gMCu3UD8RSkXCoXLfnh9n/GZ1VBxCzQT7\nqGqA2tYB4Pmh8Ruskb6FPAxWeDWM4UYzHxLWVOPcM3Bkbyt1jF3UeJ598MHvHT2F\nF4ONmuUWkBwAfJ6St3KFUjk=\n-----END PRIVATE KEY-----\n",
  "client_email": "chatbot@stellar-depth-419411.iam.gserviceaccount.com",
  "client_id": "113818006855998040942",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/chatbot%40stellar-depth-419411.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

credentials = service_account.Credentials.from_service_account_info(credentials_info)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "stellar-depth-419411-f072fec7d927.json"
genai.configure(api_key="AIzaSyALAJkf3rKlp9kagLpanYb2ZWXdHn-aOKE")






# Function to extract text from a specified PDF file
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save the FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load the conversational chain for answering questions
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Initialize the chat interface
def chatbot_response(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )
    return response["output_text"]

# Main function for chatbot interface
def main():
    st.set_page_config("JNTU BOT")
    st.header("JNTU BOT 🏛️")

    # Processing the static PDF document on startup
    pdf_path = "jntu.pdf"  # Specify the path to your static PDF document
    with st.spinner("Processing the document..."):
        raw_text = get_pdf_text(pdf_path)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("Document processed and vector store created successfully.")

    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # User input through chat interface
    prompt = st.chat_input("Ask your question here...")
    if prompt:
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate assistant response
        response = chatbot_response(prompt)
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
