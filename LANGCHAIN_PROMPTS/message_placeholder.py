from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Create prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# Store chat history
chat_history = []

# Load chat history from file
with open("chat_history.txt", "r") as f:
    for line in f:
        line = line.strip()

        if line.startswith("Human:"):
            chat_history.append(
                HumanMessage(
                    content=line.replace("Human:", "").strip()
                )
            )

        elif line.startswith("AI:"):
            chat_history.append(
                AIMessage(
                    content=line.replace("AI:", "").strip()
                )
            )

# Create final prompt
prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "Where is my refund?"
})

# Print prompt
print(prompt)

