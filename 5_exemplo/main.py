import ollama
llm = "qwen2.5"

stream = ollama.generate(model=llm, prompt='''what time is it?''', stream=True)
for chunk in stream:
    print(chunk['response'], end='', flush=True)