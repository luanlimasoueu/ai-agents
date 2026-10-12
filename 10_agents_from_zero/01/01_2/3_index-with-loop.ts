import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

const MODEL = "qwen2.5-coder:7b";
const NUM_PREDICT = 1024;

type Message = {
  role: "user" | "assistant";
  content: string;
};

async function ollama(messages: Message[]): Promise<string> {
  const response = await fetch(
    "http://127.0.0.1:11434/api/chat",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        messages,
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

  return data.message.content;
}

async function main() {
  const terminal = createInterface({
    input,
    output,
  });

  const messages: Message[] = [];

  console.log("Conversa iniciada. Digite 'sair' para encerrar.\n");

  while (true) {
    const pergunta = await terminal.question("Você: ");

    if (pergunta.toLowerCase() === "sair") {
      break;
    }

    messages.push({
      role: "user",
      content: pergunta,
    });

    try {
      const resposta = await ollama(messages);

      messages.push({
        role: "assistant",
        content: resposta,
      });

      console.log(`\nOllama: ${resposta}\n`);
    } catch (erro) {
      console.error("Erro:", erro);
    }
  }

  terminal.close();
  console.log("Conversa encerrada.");
}

main().catch(console.error);