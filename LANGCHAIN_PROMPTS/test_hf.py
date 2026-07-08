from huggingface_hub import InferenceClient

print("Starting...")

TOKEN = "your_token_here"
client = InferenceClient(api_key=TOKEN)

try:
    print("Sending request...")

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "user", "content": "Hello"}
        ],
        max_tokens=50
    )

    print("Response received")
    print(response.choices[0].message.content)

except Exception as e:
    print("ERROR TYPE:", type(e).__name__)
    print("ERROR:", str(e))