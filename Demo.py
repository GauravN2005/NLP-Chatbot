from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()  # This reads .env

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError(" GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=gemini_api_key)

# Test with the stable gemini-2.0-flash model
contents = [
    types.Content(
        role="user",
        parts=[
            types.Part.from_text(text="What is AI?"),
        ],
    ),
]

generate_content_config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
        thinking_level="HIGH",
    ),
)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=contents,
    config=generate_content_config,
)
print(response.text)
