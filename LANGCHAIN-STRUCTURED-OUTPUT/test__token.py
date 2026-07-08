from dotenv import load_dotenv
import os

load_dotenv()

print("Token =", os.getenv("your_token_here"))