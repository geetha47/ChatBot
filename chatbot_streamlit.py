import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import faiss
from sentence_transformers import SentenceTransformer
import ollama

def load_and_tokenize(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def load_retriever_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

def build_faiss_index(text_content, retriever_model):
    lines = text_content.split('. ')
    line_embeddings = retriever_model.encode(lines, convert_to_tensor=True)
    index = faiss.IndexFlatIP(line_embeddings.shape[1])
    faiss.normalize_L2(line_embeddings.cpu().detach().numpy())
    index.add(line_embeddings.cpu().detach().numpy())
    return index, lines

def retrieve_relevant_passages_with_context(query, index, preprocessed_lines, retriever_model, k=6, context_size=15):
    query_embedding = retriever_model.encode([query], convert_to_tensor=True)
    faiss.normalize_L2(query_embedding.cpu().detach().numpy())
    _, relevant_indices = index.search(query_embedding.cpu().detach().numpy(), k)
    relevant_index = relevant_indices[0][0]
    start_index = max(relevant_index - context_size, 0)
    end_index = min(relevant_index + context_size + 1, len(preprocessed_lines))
    context_lines = preprocessed_lines[start_index:end_index]
    return " ".join(context_lines)

def generate_answer_with_ollama(relevant_text, query):
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful chatbot assistant that answers questions based only on the provided text."),
        ("human", "Answer the following question: {query} based on this text:\n\nText:\n{text}"),
    ])
    messages = chat_template.format_messages(text=relevant_text, query=query)
    llm = ChatOllama(
        model="gemma:2b-instruct-v1.1-q3_K_M",
        temperature=0.1
    )
    ai_msg = llm.invoke(messages)
    return ai_msg.content

def get_response(query, file_path='dataset_latest.txt', k=6, context_size=15):
    ollama.pull('gemma:2b-instruct-v1.1-q3_K_M')
    text_content = load_and_tokenize(file_path)
    retriever_model = load_retriever_model()
    index, preprocessed_lines = build_faiss_index(text_content, retriever_model)
    relevant_context = retrieve_relevant_passages_with_context(query, index, preprocessed_lines, retriever_model, k, context_size)
    response = generate_answer_with_ollama(relevant_context, query)
    return response

# Streamlit Integration
st.set_page_config(page_title="University West Chatbot", page_icon="🎓", layout="centered")
st.markdown(
    """
    <style>
        .main {
            background-image: url('https://www.hv.se/assets/img/framework/hv-logo-small-new.png');
            background-size: cover;
            background-position: center;
            color: #424242;
            font-family: 'Arial', sans-serif;
        }
        .stTextInput, .stButton > button {
            border-radius: 10px;
            border: 1px solid #0046a1;
        }
        .stTextInput input {
            padding: 10px;
        }
        .stButton > button {
            padding: 10px;
            background-color: #00757e;
            color: white;
        }
        .user-message {
            background-color: #d3d3d3;
            padding: 10px 30px;
            border-radius: 10px;
            margin-bottom: 5px;
            margin-left: 40px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎓 University West Chatbot")
st.write("Welcome to the University West Chatbot! Ask any questions and receive relevant answers instantly.")

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

user_query = st.text_input("Ask me your question:")

if st.button("💬 Get Answer"):
    if user_query:
        with st.spinner("Thinking..."):
            response = get_response(user_query)
        st.session_state.conversation_history.append({"question": user_query, "answer": response})

for chat in st.session_state.conversation_history:
    st.markdown(f'<div class="user-message">{chat["question"]}</div>', unsafe_allow_html=True)
    st.markdown(f"{chat['answer']}")

st.sidebar.title("ℹ️ About")
st.sidebar.info("This chatbot uses a Retrieval-Augmented Generation (RAG) system with the Gemma model and FAISS for document retrieval.")
