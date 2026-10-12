'''import os
from openai import OpenAI
from dotenv import load_dotenv

# Carregando as variáveis de ambiente do arquivo .env
load_dotenv()
# Acessando a variável de ambiente API_KEY
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()  # lê OPENAI_API_KEY do ambiente

response = client.responses.create(
    model="gpt-5.2",          # troque pelo modelo atual da sua conta
    max_output_tokens=1024,
    input=[
        {
            "role": "user",
            "content": "Hello, GPT",
        }
    ],
)

for item in response.output:
    if item.type == "message":
        for part in item.content:
            if part.type == "output_text":
                print(part.text)'''

import ollama

response = ollama.chat(
    model="qwen2.5-coder:7b",
    messages=[
        {
            "role": "user",
            "content": "Hello, Ollama",
        }
    ],
    options={"num_predict": 1024},  # equivale a max_tokens
)

print(response.message.content)