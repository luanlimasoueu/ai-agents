import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { readFile } from "node:fs/promises";
import { PDFParse } from "pdf-parse";

import {
  Annotation,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";

const MODEL = "qwen2.5-coder:7b";

const State = Annotation.Root({
  caminhoPdf: Annotation<string>,
  pergunta: Annotation<string>,

  textoPdf: Annotation<string>({
    default: () => "",
  }),

  resposta: Annotation<string>({
    default: () => "",
  }),

  critica: Annotation<string>({
    default: () => "",
  }),
});

async function lerPdf(caminho: string): Promise<string> {
  const arquivo = await readFile(caminho);

  const parser = new PDFParse({
    data: arquivo,
  });

  try {
    const resultado = await parser.getText();

    return resultado.text;
  } finally {
    await parser.destroy();
  }
}

async function chamarOllama(
  system: string,
  prompt: string
): Promise<string> {
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
            role: "system",
            content: system,
          },
          {
            role: "user",
            content: prompt,
          },
        ],
        stream: false,
        options: {
          num_predict: 512,
          temperature: 0.2,
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

async function leitorPdf(state: typeof State.State) {
  console.log("\n[1/3] Lendo o PDF...");

  const textoPdf = await lerPdf(state.caminhoPdf);

  if (!textoPdf.trim()) {
    throw new Error(
      "Não foi possível extrair texto deste PDF."
    );
  }

  console.log(
    `Texto extraído: ${textoPdf.length} caracteres`
  );

  return {
    textoPdf,
  };
}

async function agenteResposta(
  state: typeof State.State
) {
  console.log("\n[2/3] Agente escrevendo a resposta...");

  const resposta = await chamarOllama(
    "Você é um agente especialista em leitura de documentos. Responda usando somente o conteúdo do PDF.",
    `
CONTEÚDO DO PDF:
${state.textoPdf}

PERGUNTA:
${state.pergunta}

Escreva uma resposta clara e objetiva.
Se a informação não estiver no PDF, diga que não encontrou.
`
  );

  console.log("\n--- RESPOSTA DO AGENTE ---");
  console.log(resposta);

  return {
    resposta,
  };
}

async function agenteCritico(
  state: typeof State.State
) {
  console.log("\n[3/3] Agente crítico analisando...");

  const critica = await chamarOllama(
    "Você é um agente crítico. Verifique se a resposta está correta de acordo com o PDF.",
    `
CONTEÚDO DO PDF:
${state.textoPdf}

PERGUNTA:
${state.pergunta}

RESPOSTA DO PRIMEIRO AGENTE:
${state.resposta}

Analise:
1. Se a resposta está baseada no PDF.
2. Se existem erros.
3. Se faltou alguma informação.

Se estiver correta, diga isso claramente.
`
  );

  console.log("\n--- CRÍTICA ---");
  console.log(critica);

  return {
    critica,
  };
}

const fluxo = new StateGraph(State)
  .addNode("leitorPdf", leitorPdf)
  .addNode("agenteResposta", agenteResposta)
  .addNode("agenteCritico", agenteCritico)
  .addEdge(START, "leitorPdf")
  .addEdge("leitorPdf", "agenteResposta")
  .addEdge("agenteResposta", "agenteCritico")
  .addEdge("agenteCritico", END)
  .compile();

async function main() {
  const terminal = createInterface({
    input,
    output,
  });

  const caminhoPdf = await terminal.question(
    "Caminho do PDF: "
  );

  const pergunta = await terminal.question(
    "Faça sua pergunta: "
  );

  terminal.close();

  await fluxo.invoke({
    caminhoPdf: caminhoPdf.trim(),
    pergunta: pergunta.trim(),
  });

  console.log("\n===== PROCESSO CONCLUÍDO =====");
}

main().catch((erro) => {
  console.error("\nErro:", erro);
});