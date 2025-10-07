import requests

url = "http://localhost:11434/api/generate"
pergunta = input("Digite sua pergunta: ")

#programa de chatbot que acessa a api de ollama
dados = {
    "model": 'llama3.2:3b',
    "prompt":pergunta,
    "stream": False
}

resposta =  requests.post( url, json = dados)
print( resposta.json()['response'])