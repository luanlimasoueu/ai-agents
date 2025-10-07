from ollama import chat

#programa chatbot
pergunta = input("Digite sua pergunta: ")
messages=[
    {'role': 'user', 
     'content': pergunta}
]

resposta = chat('llama3.2:3b', messages = messages)
print(resposta['message']['content'] )