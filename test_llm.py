import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_groq(prompt: str):
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# Guarded so importing this file elsewhere never fires a real API call as a
# side effect (the previous version ran this at module import time).
if __name__ == "__main__":
    answer = ask_groq("Explain what an AI agent is in simple words.")
    print(answer)
