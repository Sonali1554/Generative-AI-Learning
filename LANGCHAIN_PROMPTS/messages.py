from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN")
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]

hf_messages = []

for msg in messages:
    if isinstance(msg, SystemMessage):
        hf_messages.append(
            {
                "role": "system",
                "content": msg.content
            }
        )

    elif isinstance(msg, HumanMessage):
        hf_messages.append(
            {
                "role": "user",
                "content": msg.content
            }
        )

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=hf_messages,
    max_tokens=100
)

result = response.choices[0].message.content

messages.append(
    AIMessage(content=result)
)

print(messages)