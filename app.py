import streamlit as st
from streamlit_chat import message
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model_path = './finetuned_model'
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

def generate_response(input_text):
    # Encode the text prompt
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    # Generate text using the model
    max_length = 50  # Adjust as needed
    sample_output = model.generate(
        input_ids,
        do_sample=True,
        max_length=max_length,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )[0]
    # Decode and return the generated text
    text = tokenizer.decode(sample_output, skip_special_tokens=True)
    text = text.split('.')
    text = text[0] + '.'
    return text

st.title("Sun Tzu Generator")

os.environ['CURL_CA_BUNDLE'] = '' # per https://stackoverflow.com/a/75746105

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
with st.chat_message("assistant"):
    st.write("Hi, How can I assist you?")
    
if prompt := st.chat_input("Your question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            message_placeholder = st.empty()
            full_response = generate_response(prompt)
            message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
