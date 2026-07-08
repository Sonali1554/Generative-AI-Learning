from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Load Hugging Face Model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    task="text-generation",
    max_new_tokens=100
)

# Create Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}."
)

# Create Chain
chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# Run Chain
topic = input("Enter a topic: ")

output = chain.run(topic)

print("\nGenerated Blog Title:")
print(output)