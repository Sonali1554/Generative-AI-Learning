from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables
load_dotenv()

# ====================================
# Define Tool
# ====================================

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers.
    """
    return a * b


# ====================================
# Hugging Face LLM
# ====================================

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.2,
    max_new_tokens=512
)

chat_model = ChatHuggingFace(llm=llm)

# Bind tools
llm_with_tools = chat_model.bind_tools([multiply])

# ====================================
# User Question
# ====================================

messages = [
    HumanMessage(
        content="What is 45 multiplied by 12?"
    )
]

# ====================================
# First LLM Call
# ====================================

response = llm_with_tools.invoke(messages)

print("\nTool Call Generated:")
print(response.tool_calls)

messages.append(response)

# ====================================
# Execute Tool
# ====================================

for tool_call in response.tool_calls:

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name == "multiply":

        result = multiply.invoke(tool_args)

        print("\nTool Result:")
        print(result)

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

# ====================================
# Final LLM Response
# ====================================

final_response = llm_with_tools.invoke(messages)

print("\nFinal Answer:")
print(final_response.content)