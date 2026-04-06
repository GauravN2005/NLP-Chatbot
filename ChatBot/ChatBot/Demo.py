from google import genai
from dotenv import load_dotenv
import os

load_dotenv()  # This reads .env

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("⚠️ GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=gemini_api_key)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="What is AI?"
)
print(response.text)
