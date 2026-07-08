from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant that provides information about cricket."),
    HumanMessage(content="Tell me about the cricket shot called '{topic}' in the domain of '{domain}'.")
])

prompt = chat_template.invoke({
    "domain": "cricket",
    "topic": "Dusra"
})

print(prompt)