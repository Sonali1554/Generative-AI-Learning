from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Get token from .env
token = os.getenv("HF_TOKEN")

if not token:
    print("HF_TOKEN not found in .env file")
    exit()

# Create Hugging Face client
client = InferenceClient(
    api_key=token
)

print("🤖 Chatbot Started! Type 'exit' to quit.")

# Store complete conversation history
chat_history = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    }
]

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # Save user message
    chat_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    try:

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=chat_history,
            max_tokens=300
        )

        ai_response = response.choices[0].message.content

        print("AI:", ai_response)

        # Save assistant response
        chat_history.append(
            {
                "role": "assistant",
                "content": ai_response
            }
        )

    except Exception as e:
        print("Error:", e)

# Save chat history when exiting
with open("chat_history.json", "w", encoding="utf-8") as file:
    json.dump(chat_history, file, indent=4, ensure_ascii=False)

print("\n✅ Chat history saved to chat_history.json")
print("👋 Goodbye!")