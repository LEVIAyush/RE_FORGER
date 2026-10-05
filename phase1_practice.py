import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm import call_llm


# Exercise 1 – Role Definition

# system_prompt = """
# You are a personal assistant that is helpful, creative, clever, and very friendly.
# your job is to give reality chechk  to  user and make them realises their mistakes about themselves."""

# user_prompt = "I am inevitable "

# messages = [
#     {"role": "system", "content": system_prompt},
#     {"role": "user", "content": user_prompt}
# ]

# response = call_llm(messages)
# print(response)



#  Chain-of-Thought Prompting
# Exercise 2 – Force Step-by-Step Thinking

system_prompt = """
You are a careful researcher.
Always think step by step before answering.
"""

user_prompt = """
What are the main differences between Attack titan and founding titan?
Think step by step.
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

# print(call_llm(messages))






# Structured JSON Output (Very Important)
# Exercise 3 – Force JSON

system_prompt = """
You are a research assistant. you talf about only sientific researches in feild of science only .
if you are asked about any other topic you must say "I am sorry, I can only provide information on scientific research topics."
You must always reply in valid JSON format only.
Do not write any extra text outside the JSON.
"""

user_prompt = """
Analyze the topic "Titan shifters","Who was right in AOT", "do you consider AOT The best anime of all time" and return this JSON structure:

{
  "definition": "...",
  "key_characteristics": ["...", "..."],
  "examples": ["...", "..."],
  "difficulty_level": "Beginner/Intermediate/Advanced"
}
the definition should be a brief summary of the topic.
create a diffrent json structure for each topic and return them in a list.
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

response = call_llm(messages)
print(response)






# Parsing

# import json

# try:
#     data = json.loads(response)
#     print("Successfully parsed JSON!")
#     print(data["definition"])
# except json.JSONDecodeError:
#     print("Failed to parse JSON. The model broke the format.")