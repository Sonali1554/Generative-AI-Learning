from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Optional, Literal
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):

    key_themes: Annotated[
        list[str],
        "Write down all the key themes discussed in the review in a list"
    ]

    summary: Annotated[
        str,
        "A brief summary of the review"
    ]

    sentiment: Annotated[
        Literal["pos", "neg", "neu"],
        "Return sentiment of the review either positive, negative or neutral"
    ]

    pros: Annotated[
        Optional[list[str]],
        "Write down all the pros inside a list"
    ]

    cons: Annotated[
        Optional[list[str]],
        "Write down all the cons inside a list"
    ]

    name: Annotated[
        Optional[str],
        "Write the name of the reviewer"
    ]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say,
it's an absolute powerhouse! The Snapdragon processor makes everything
lightning fast. The camera quality is excellent and battery life is amazing.
However, the phone is a bit heavy and expensive.
My name is Sonali.
""")

print(result)