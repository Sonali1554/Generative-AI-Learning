from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create Hugging Face LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    task="text-generation",
    max_new_tokens=512
)

# Create Chat Model
model = ChatHuggingFace(llm=llm)

# Prompt 1: Generate Detailed Report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"]
)

# Prompt 2: Generate Summary
template2 = PromptTemplate(
    template="Write a concise 5-line summary of the following text:\n\n{text}",
    input_variables=["text"]
)

# Generate first prompt
prompt1 = template1.invoke({
    "topic": "Black Holes"
})

# Generate detailed report
report = model.invoke(prompt1)

# Generate second prompt using report
prompt2 = template2.invoke({
    "text": report.content
})

# Generate summary
summary = model.invoke(prompt2)

# Print outputs
print("=" * 50)
print("DETAILED REPORT")
print("=" * 50)
print(report.content)

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(summary.content)