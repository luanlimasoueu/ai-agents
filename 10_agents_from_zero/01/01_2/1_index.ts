const MODEL = "qwen2.5-coder:7b";
const NUM_PREDICT = 1024;

async function ollama() {
  const response = await fetch(
    "http://127.0.0.1:11434/api/chat",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          {
            role: "user",
            content: "Hello, Ollama",
          },
        ],
        stream: false,
        options: {
          num_predict: NUM_PREDICT,
        },
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`Erro no Ollama: ${response.status}`);
  }

  const data = await response.json();

  console.log(data.message.content);
}

ollama().catch(console.error);