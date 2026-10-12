async function main() {
  console.log("Enviando pergunta para o Ollama...");

  const response = await fetch(
    "http://127.0.0.1:11434/api/chat",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "qwen2.5-coder:7b",
        messages: [
          {
            role: "user",
            content: "Responda apenas: Ollama funcionando",
          },
        ],
        stream: false,
      }),
    }
  );

  console.log("Status HTTP:", response.status);

  const data = await response.json();

  console.log("Resposta:");
  console.log(data.message?.content);
}

main().catch((erro) => {
  console.error("Erro:", erro);
});