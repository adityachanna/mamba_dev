from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
load_dotenv()
import os


load_dotenv()


model = ChatOpenRouter(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    temperature=0.8,
    api_key=os.getenv("OPENROUTER_API_KEY"),

)

# Example usage
response = model.invoke("What NFL team won the Super Bowl in the year Justin Bieber was born?")
print(response.content)
