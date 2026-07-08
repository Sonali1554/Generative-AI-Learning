from langchain_core.prompts import PromptTemplate

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

template.save('template.json')



