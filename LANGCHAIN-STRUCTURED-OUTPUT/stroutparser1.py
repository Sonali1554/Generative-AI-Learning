from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    task="text-generation",
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a concise 5-line summary of the following text:\n\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = (
    template1
    | model
    | parser
    | RunnableLambda(lambda x: {"text": x})
    | template2
    | model
    | parser
)

result = chain.invoke({
    "topic": "Black Holes"
})

print(result)