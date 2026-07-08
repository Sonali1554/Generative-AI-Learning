from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import InferenceClient
import os

load_dotenv()

st.header("Research Tool")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Select...",
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (Detailed Explanation)"
    ]
)

if st.button("Summarize"):

    prompt = f"""
    Explain the research paper '{paper_input}'.

    Explanation Style: {style_input}

    Explanation Length: {length_input}

    Provide a clear explanation.
    """

    client = InferenceClient(
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    response = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    st.subheader("Summary")
    st.write(response.choices[0].message.content)