import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("Waris01/google-t5-finetuning-text-summarization")
tokenizer = T5Tokenizer.from_pretrained("Waris01/google-t5-finetuning-text-summarization")

st.title("Text Summarization App")
st.write("Enter the text you want to summarize:")

# Text input from the user
input_text = st.text_area("Input Text", height=300)

# Button to generate summary
if st.button("Generate Summary"):
    if input_text:
        # Tokenize the input text
        inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)

        # Generate the summary
        summary_ids = model.generate(inputs, max_length=130, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.subheader("Generated Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")

st.markdown("Made with ❤️ using Hugging Face Transformers and Streamlit.")
