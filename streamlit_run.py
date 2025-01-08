import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Paths to the BERT model and tokenizer
MODEL_PATH = "D:\git\metaai\enhanced_riddle_model.joblib"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit app UI
st.title("Riddle Solver (BERT Model)")
st.write("Type a riddle below, and the BERT model will try to predict the answer!")

# Input riddle
user_riddle = st.text_area("Enter a riddle:", placeholder="Type your riddle here...")

if st.button("Get Answer"):
    if user_riddle.strip():
        # Tokenize and prepare input for the model
        inputs = tokenizer(user_riddle, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
        
        # Map predicted class to the corresponding answer (update this mapping if needed)
        # Assuming you have class-to-answer mapping saved somewhere
        class_to_answer = {
            0: "An echo",
            1: "Footsteps",
            2: "A map",
            3: "A keyboard",
            4: "A candle",
            5: "A coin",
            6: "A bank"
        }
        
        predicted_answer = class_to_answer.get(predicted_class, "Unknown")
        
        # Display the result
        st.subheader("Predicted Answer:")
        st.success(predicted_answer)
    else:
        st.error("Please enter a riddle.")

# Footer
st.write("---")
st.write("Enhanced BERT Model | Powered by Streamlit")
