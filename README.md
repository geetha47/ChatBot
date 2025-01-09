##University Chatbot Project##

This project involves developing an AI-powered chatbot designed to help university students find the information they need easily and efficiently. 🎓💬

1. Data Collection, Cleaning & Preprocessing

Collected data using Beautiful Soup through web scraping.

Removed duplicates, trailing whitespaces, and converted text to lowercase for consistency.

Preprocessed with NLTK: stopword removal and tokenization.

2. Enhancing with RAG

Implemented Retrieval-Augmented Generation (RAG) inspired by 'Building LLMs for Production' by Louis-Francois Bouchard and Louie Peters.

RAG improved results by retrieving the most relevant information before passing it to the language model, reducing misinformation.

3. Final Model Choice

Tested Google's Gemma-2 2B model and found it efficient, accurate, and with minimal hallucinations.

Compared both the standard 2B and the quantized version, ultimately choosing the standard 2B for its higher accuracy.

4. Technical Setup

Rented GPUs through RunPod for model fine-tuning and testing.

Deployed the chatbot as a Streamlit web app for demonstration purposes (not publicly accessible for students).
