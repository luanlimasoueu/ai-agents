import {
  Annotation,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";

const State = Annotation.Root({
  pergunta: Annotation<string>,
  resposta: Annotation<string>,
});

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

  console.log("Status:", response.status);

  if (!response.ok) {
    throw new Error(`Erro no Ollama: ${response.status}`);
  }

  const data = await response.json();

  console.log("Resposta recebida.");

  return data.message.content;
}

const fluxo = new StateGraph(State)
  .addNode("responder", async (state) => {
    console.log("Executando nó responder...");

    const resposta = await chamarOllama(state.pergunta);

    return {
      resposta,
    };
  })
  .addEdge(START, "responder")
  .addEdge("responder", END)
  .compile();

async function main() {
  console.log("Iniciando fluxo...");

  const resultado = await fluxo.invoke({
    pergunta: "O que é um agente de IA?",
  });

  console.log("\nResposta final:");
  console.log(resultado.resposta);
}

main().catch((erro) => {
  console.error("Erro:", erro);
});