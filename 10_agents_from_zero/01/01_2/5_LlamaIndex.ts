import {
  createWorkflow,
  workflowEvent,
} from "@llama-flow/core";

const perguntaEvent = workflowEvent<string>();
const respostaEvent = workflowEvent<string>();

async function chamarOllama(pergunta: string): Promise<string> {
  console.log("Chamando Ollama...");

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
            content: pergunta,
          },
        ],
        stream: false,
        options: {
          num_predict: 256,
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

const fluxo = createWorkflow();

fluxo.handle([perguntaEvent], async (event) => {
  const resposta = await chamarOllama(event.data);

  return respostaEvent.with(resposta);
});

async function main() {
  const { stream, sendEvent } = fluxo.createContext();

  sendEvent(
    perguntaEvent.with("O que é um agente de IA?")
  );

  for await (const event of stream) {
    if (respostaEvent.include(event)) {
      console.log("\nResposta:");
      console.log(event.data);
      break;
    }
  }
}

main().catch((erro) => {
  console.error("Erro:", erro);
});