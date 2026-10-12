from pathlib import Path

from pypdf import PdfReader
from crewai import Agent, Crew, LLM, Process, Task


PDF_PATH = Path("Modelo-de-contrato-de-aluguel-1.pdf")
MODEL = "ollama/qwen2.5-coder:7b"


def ler_pdf(caminho: Path) -> str:
    leitor = PdfReader(caminho)

    texto = ""

    for pagina in leitor.pages:
        texto += pagina.extract_text() or ""

    return texto


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF não encontrado: {PDF_PATH}"
        )

    print("[1/3] Lendo o PDF...")

    texto_pdf = ler_pdf(PDF_PATH)

    if not texto_pdf.strip():
        raise ValueError(
            "Não foi possível extrair texto do PDF."
        )

    print(
        f"PDF lido com sucesso: "
        f"{len(texto_pdf)} caracteres"
    )

    pergunta = input(
        "\nFaça uma pergunta sobre o PDF: "
    )

    modelo = LLM(
        model=MODEL,
        base_url="http://127.0.0.1:11434",
        temperature=0.2,
        max_tokens=512,
    )

    agente_leitor = Agent(
        role="Leitor de PDF",
        goal="Responder perguntas usando somente o conteúdo do PDF",
        backstory=(
            "Você analisa documentos e responde "
            "de forma clara e objetiva."
        ),
        llm=modelo,
        verbose=True,
    )

    agente_critico = Agent(
        role="Crítico da resposta",
        goal="Verificar se a resposta está correta",
        backstory=(
            "Você confere se uma resposta está baseada "
            "corretamente no documento."
        ),
        llm=modelo,
        verbose=True,
    )

    print("\n[2/3] Agente leitor escrevendo a resposta...")

    tarefa_resposta = Task(
        description=f"""
Leia o conteúdo do PDF e responda à pergunta.

CONTEÚDO DO PDF:
{texto_pdf}

PERGUNTA:
{pergunta}

Regras:
- Use somente o conteúdo do PDF.
- Não invente informações.
- Se não encontrar a resposta, diga isso.
- Seja claro e objetivo.
""",
        expected_output=(
            "Uma resposta clara e baseada no conteúdo do PDF."
        ),
        agent=agente_leitor,
    )

    crew_leitor = Crew(
        agents=[agente_leitor],
        tasks=[tarefa_resposta],
        process=Process.sequential,
        verbose=True,
    )

    resultado_resposta = crew_leitor.kickoff()
    resposta = resultado_resposta.raw

    print("\n===== RESPOSTA DO AGENTE LEITOR =====")
    print(resposta)

    print("\n[3/3] Agente crítico analisando...")

    tarefa_critica = Task(
        description=f"""
Analise a resposta produzida pelo agente leitor.

PERGUNTA:
{pergunta}

CONTEÚDO DO PDF:
{texto_pdf}

RESPOSTA DO AGENTE LEITOR:
{resposta}

Verifique:

1. A resposta está baseada no PDF?
2. Existem erros?
3. Faltou alguma informação importante?
4. O agente inventou alguma informação?

Faça uma crítica objetiva.
""",
        expected_output=(
            "Uma crítica clara sobre a qualidade e "
            "a fidelidade da resposta."
        ),
        agent=agente_critico,
    )

    crew_critico = Crew(
        agents=[agente_critico],
        tasks=[tarefa_critica],
        process=Process.sequential,
        verbose=True,
    )

    resultado_critica = crew_critico.kickoff()
    critica = resultado_critica.raw

    print("\n===== CRÍTICA DO SEGUNDO AGENTE =====")
    print(critica)

    print("\n===== PROCESSO CONCLUÍDO =====")


if __name__ == "__main__":
    main()