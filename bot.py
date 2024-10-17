#!/usr/bin/env python
# coding: utf-8

# In[1]:


import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
import os
from huggingface_hub import login 


# In[38]:


#!pip install streamlit


# In[39]:


import streamlit as st


# In[1]:


#!pip install transformers datasets faiss-gpu torch sentence_transformers


# In[2]:


#!pip install accelerate==1.0


# In[40]:


# Step 1: Load and Preprocess Dataset (split by lines instead of paragraphs)
@st.cache_resource
def load_and_tokenize(file_path):
    with open(file_path, 'r') as file:
        text = file.read()  # Read the entire content of the file
    # Split the text into lines
    lines = text.split('. ')  # Assuming period as line separator, adjust as needed
    return lines


# In[4]:


file_path = 'hv_english3.txt'
preprocessed_lines = load_and_tokenize(file_path)


# In[42]:


@st.cache_resource
def load_retriever_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')


# In[43]:


retriever_model = load_retriever_model()


# In[6]:





# In[47]:


@st.cache_resource
def build_faiss_index(preprocessed_lines):
    line_embeddings = retriever_model.encode(preprocessed_lines, convert_to_tensor=True)
    index = faiss.IndexFlatL2(line_embeddings.shape[1])
    faiss.normalize_L2(line_embeddings.cpu().detach().numpy())
    index.add(line_embeddings.cpu().detach().numpy())
    return index

index = build_faiss_index(preprocessed_lines)


# In[7]:





# In[48]:


def retrieve_relevant_passages_with_context(query, k=3, context_size=15):
    query_embedding = retriever_model.encode([query], convert_to_tensor=True)
    faiss.normalize_L2(query_embedding.cpu().detach().numpy())
    _, relevant_indices = index.search(query_embedding.cpu().detach().numpy(), k)
    relevant_index = relevant_indices[0][0]
    start_index = max(relevant_index - context_size, 0)
    end_index = min(relevant_index + context_size + 1, len(preprocessed_lines))
    context_lines = preprocessed_lines[start_index:end_index]
    return " ".join(context_lines)


# In[47]:


# Step 5: Load LLaMA model for information extraction
#model_id = "AI-Sweden-Models/Llama-3-8B-instruct"


# In[ ]:





# In[9]:


hf_token = os.getenv("HUGGINGFACE_TOKEN")



# In[ ]:


# Login using the token
if hf_token:
    login(token=hf_token)


# In[10]:


model_id ='meta-llama/Llama-3.1-8B-Instruct'


# In[45]:


@st.cache_resource
def load_llama_model():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    return pipeline("text-generation", model=model_id, tokenizer=model_id, torch_dtype=torch.bfloat16, device_map="auto")


# In[46]:


llama_pipeline = llama_pipeline = load_llama_model()


# In[32]:


def correct_text_with_llama(text, query, max_new_tokens):
    # Combine text and query into messages suitable for the LLaMA model
    messages = [
        {"role": "system", "content": "You are an expert chatbot assistant who answers specific questions analysing only the text provided to you.Please provide a concise answer without any formalities like 'Best regards' or repeated suggestions. "}, 
        {"role": "user", "content": f"Answer this question: {query}, from this text: \"{text}\""}
    ]

    # Define terminators for generation stopping
   #terminators = [
   #       llama_pipeline.tokenizer.eos_token_id,
   #       llama_pipeline.tokenizer.convert_tokens_to_ids("")
   #]
    eos_token_id = llama_pipeline.tokenizer.eos_token_id
    terminators = eos_token_id
    repetition_penalty=1.2
    # Generate output from the LLaMA model
    output = llama_pipeline(
        messages,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.8,
        repetition_penalty=repetition_penalty
    )

    # Extract the assistant's reply from the output
    return output[0]['generated_text']



# In[33]:


# Step 6: Generate Response by First Retrieving Relevant Context, Then Using LLaMA for Extraction
def generate_response(query, k, context_size):
    # Step 1: Retrieve the most relevant passages with 10 lines before and after
    relevant_context = retrieve_relevant_passages_with_context(query, k, context_size)
    print(relevant_context)
    # Step 2: Pass the context to LLaMA to extract the specific relevant information
    extracted_information = correct_text_with_llama(relevant_context, query, max_new_tokens=4096)
    
    # Return the extracted information as the final response
    return extracted_information


# In[49]:


st.title("University Chatbot")


# In[50]:


query = st.text_input("Ask me a question:", "")


# In[51]:


if st.button("Submit"):
    if query:
        with st.spinner("Searching..."):
            response = generate_response(query, k=5, context_size=10)
        st.write(f"Chatbot Response: {response}")
    else:
        st.write("Please enter a valid question.")


# In[ ]:





# In[ ]:




