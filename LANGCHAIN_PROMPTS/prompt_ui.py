from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate,load_prompt
import os

load_dotenv()

st.set_page_config(page_title="Research Tool", page_icon="📚")

st.header("📚 Research Paper Summarizer")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
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

template = load_prompt('template.json')
#template
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper}".

Explanation Style: {style}

Explanation Length: {length}

Requirements:

1. Mathematical Details:
   - Include relevant mathematical equations if present.
   - Explain mathematical concepts in simple language.

2. Analogies:
   - Use relatable analogies to simplify complex ideas.

3. Applications:
   - Mention practical applications and real-world use cases.

4. Key Contributions:
   - Highlight the main innovations introduced by the paper.

5. Limitations:
   - Mention important limitations if applicable.

If sufficient information is unavailable, respond with:
"Insufficient information available."
""",
    input_variables=["paper", "style", "length"]
)

if st.button("Summarize"):

    final_prompt = template.format(
        paper=paper_input,
        style=style_input,
        length=length_input
    )

    try:
        client = InferenceClient(
            api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )

        response = client.chat_completion(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            max_tokens=1000
        )

        st.subheader("📄 Summary")
        st.write(response.choices[0].message.content)

    except Exception as e:
        st.error(f"Error: {e}")