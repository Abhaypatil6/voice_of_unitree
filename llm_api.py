import os
from dotenv import load_dotenv
from google import genai

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)
MODEL = "gemini-3.8-flash"


def get_reply(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL,
        contents=(
            "You are a friendly Unitree G1 robot voice assistant. "
            "Keep replies natural and concise for spoken conversation.\n\n"
            f"User: {prompt}"
        ),
    )

    if not response.text:
        raise RuntimeError("The model returned an empty response.")

    return response.text.strip()


if __name__ == "__main__":
    try:
        prompt = input("You: ").strip()
        if prompt:
            print("G1 assistant:", get_reply(prompt))
    except Exception as exc:
        print(f"API error: {exc}")
