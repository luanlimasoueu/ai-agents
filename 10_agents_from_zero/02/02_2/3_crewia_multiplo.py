"""orquestrador inicial
        ↓
dois leitores em paralelo
        ↓
consenso
        ↓
crítico
        ↓
orquestrador final"""

import asyncio
from pathlib import Path

from pydantic import BaseModel
from pypdf import PdfReader

from crewai import Agent, LLM
from crewai.flow.flow import Flow, listen, start


MODEL = "qwen2.5-coder:7b"
MAX_TOKENS = 1024
PDF_PADRAO = "Modelo-de-contrato-de-aluguel-1.pdf"


class EstadoPDF(BaseModel):
    texto_pdf: str = ""
    pergunta: str = ""
    leitura_1: str = ""
    leitura_2: str = ""
    consenso: str = ""
    critica: str = ""
    resposta_final: str = ""


def mostrar(texto: str = "") -> None:
    print(texto, flush=True)


def extrair_texto_pdf(caminho: Path) -> str:
    leitor = PdfReader(str(caminho))
    paginas = []

    for numero, pagina in enumerate(leitor.pages, start=1):
        texto = pagina.extract_text() or ""
        paginas.append(f"\n--- PÁGINA {numero} ---\n{texto}")

    texto_completo = "\n".join(paginas).strip()

    if not texto_completo:
        raise ValueError(
            "Não foi possível extrair texto do PDF. "
            "Talvez ele seja um PDF escaneado."
        )

    return texto_completo


def criar_llm() -> LLM:
    return LLM(
        model=f"ollama/{MODEL}",
        base_url="http://127.0.0.1:11434",
        temperature=0.2,
        max_tokens=MAX_TOKENS,
    )


def criar_agente(
    role: str,
    goal: str,
    backstory: str,
) -> Agent:
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=criar_llm(),
        verbose=False,
        allow_delegation=False,
    )


async def chamar_agente(
    agente: Agent,
    prompt: str,
) -> str:
    resultado = await agente.kickoff_async(prompt)

    texto = getattr(resultado, "raw", None)

    if texto:
        return str(texto).strip()

    return str(resultado).strip()


class FluxoPDF(Flow[EstadoPDF]):

    # =========================================================
    # 1. ORQUESTRADOR INICIAL
    # =========================================================

    @start()
    async def orquestrador_inicial(self) -> str:
        mostrar("\n===================================")
        mostrar("[1/5] ORQUESTRADOR INICIAL")
        mostrar("===================================")
        mostrar("O orquestrador recebeu o PDF.")
        mostrar(f"Pergunta recebida: {self.state.pergunta}")
        mostrar("Enviando o PDF para dois leitores...\n")

        return "iniciar_leitores"

    # =========================================================
    # LEITOR 1
    # =========================================================

    async def executar_leitor_1(self) -> str:
        mostrar("[2/5] LEITOR 1 começou.")

        agente = criar_agente(
            role="Leitor objetivo",
            goal="Ler o PDF e responder objetivamente à pergunta.",
            backstory=(
                "Você é um leitor objetivo. "
                "Utiliza somente informações encontradas no PDF."
            ),
        )

        prompt = f"""
Leia o PDF e responda à pergunta do usuário.

Regras:

- não invente informações;
- use somente o conteúdo do PDF;
- se a informação não estiver no documento, diga isso;
- seja objetivo.

PERGUNTA:
{self.state.pergunta}

PDF:
{self.state.texto_pdf}
"""

        resposta = await chamar_agente(agente, prompt)

        mostrar("[2/5] LEITOR 1 terminou.")

        return resposta

    # =========================================================
    # LEITOR 2
    # =========================================================

    async def executar_leitor_2(self) -> str:
        mostrar("[2/5] LEITOR 2 começou.")

        agente = criar_agente(
            role="Leitor analítico",
            goal="Analisar o PDF procurando detalhes e exceções.",
            backstory=(
                "Você é cuidadoso e procura obrigações, "
                "condições, exceções e ambiguidades."
            ),
        )

        prompt = f"""
Analise o PDF e responda à pergunta do usuário.

Procure:

- obrigações;
- condições;
- exceções;
- detalhes importantes;
- ambiguidades.

Regras:

- não invente informações;
- use somente o conteúdo do PDF;
- seja claro e objetivo.

PERGUNTA:
{self.state.pergunta}

PDF:
{self.state.texto_pdf}
"""

        resposta = await chamar_agente(agente, prompt)

        mostrar("[2/5] LEITOR 2 terminou.")

        return resposta

    # =========================================================
    # 2. LEITORES EM PARALELO
    # =========================================================

    @listen(orquestrador_inicial)
    async def leitores_paralelos(self, _rota: str) -> str:
        mostrar("\n[2/5] DOIS LEITORES TRABALHANDO EM PARALELO")
        mostrar("O Leitor 1 e o Leitor 2 começaram juntos.\n")

        leitura_1, leitura_2 = await asyncio.gather(
            self.executar_leitor_1(),
            self.executar_leitor_2(),
        )

        self.state.leitura_1 = leitura_1
        self.state.leitura_2 = leitura_2

        mostrar("\n--- RESPOSTA DO LEITOR 1 ---")
        mostrar(leitura_1)

        mostrar("\n--- RESPOSTA DO LEITOR 2 ---")
        mostrar(leitura_2)

        mostrar("\n[ROTA] LEITORES -> CONSENSO")

        return "leituras_concluidas"

    # =========================================================
    # 3. CONSENSO
    # =========================================================

    @listen(leitores_paralelos)
    async def consenso(self, _leituras: str) -> str:
        mostrar("\n===================================")
        mostrar("[3/5] ETAPA DE CONSENSO")
        mostrar("===================================")

        mostrar("\n[3.1] O consenso recebeu a resposta do Leitor 1:")
        mostrar(self.state.leitura_1)

        mostrar("\n[3.2] O consenso recebeu a resposta do Leitor 2:")
        mostrar(self.state.leitura_2)

        mostrar("\n[3.3] Comparando as duas leituras...")

        agente = criar_agente(
            role="Mediador de consenso",
            goal="Comparar as duas leituras e produzir um consenso.",
            backstory=(
                "Você compara duas análises do mesmo documento. "
                "Separa concordâncias de divergências reais."
            ),
        )

        prompt = f"""
Compare as duas leituras abaixo.

Organize a resposta exatamente nestas partes:

## Pontos de concordância

Liste os pontos em que os dois leitores chegaram à mesma conclusão.

## Divergências reais

Liste somente os pontos em que as leituras realmente são diferentes.

Não considere divergência quando os dois leitores disserem a mesma coisa.

Se não houver divergências reais, escreva:

"Nenhuma divergência real foi encontrada."

## Consenso final

Produza uma conclusão objetiva para responder à pergunta.

Regras:

- não invente informações;
- não transforme concordâncias em divergências;
- utilize somente as informações das leituras.

PERGUNTA:
{self.state.pergunta}

LEITURA 1:
{self.state.leitura_1}

LEITURA 2:
{self.state.leitura_2}
"""

        mostrar("\n[3.4] Enviando as duas leituras ao agente de consenso...")

        resposta = await chamar_agente(agente, prompt)

        self.state.consenso = resposta

        mostrar("\n[3.5] RESULTADO DO CONSENSO:")
        mostrar(resposta)

        mostrar("\n[3.6] Consenso terminado.")
        mostrar("[ROTA] CONSENSO -> CRÍTICO")

        return resposta

    # =========================================================
    # 4. CRÍTICO
    # =========================================================

    @listen(consenso)
    async def critico(self, consenso_recebido: str) -> str:
        mostrar("\n===================================")
        mostrar("[4/5] ETAPA DO CRÍTICO")
        mostrar("===================================")

        mostrar("\n[4.1] O crítico recebeu este consenso:")
        mostrar(consenso_recebido)

        mostrar("\n[4.2] O crítico recebeu o PDF original.")

        agente = criar_agente(
            role="Agente crítico",
            goal="Verificar se o consenso está correto e fiel ao PDF.",
            backstory=(
                "Você é rigoroso. Procura erros, omissões, "
                "contradições e informações sem comprovação."
            ),
        )

        prompt = f"""
Valide o consenso comparando-o com o PDF original.

Analise:

1. O que está correto.
2. O que está incorreto.
3. O que está incompleto.
4. O que não está comprovado.
5. O que precisa ser corrigido.

Se o consenso estiver correto, escreva:

"O consenso está de acordo com o PDF."

Não invente informações.

CONSENSO:
{consenso_recebido}

PDF ORIGINAL:
{self.state.texto_pdf}
"""

        mostrar("\n[4.3] Enviando consenso e PDF ao crítico...")

        resposta = await chamar_agente(agente, prompt)

        self.state.critica = resposta

        mostrar("\n[4.4] RESULTADO DA CRÍTICA:")
        mostrar(resposta)

        mostrar("\n[4.5] Crítico terminou.")
        mostrar("[ROTA] CRÍTICO -> ORQUESTRADOR FINAL")

        return resposta

    # =========================================================
    # 5. ORQUESTRADOR FINAL
    # =========================================================

    @listen(critico)
    async def orquestrador_final(
        self,
        critica_recebida: str,
    ) -> str:
        mostrar("\n===================================")
        mostrar("[5/5] ETAPA DO ORQUESTRADOR FINAL")
        mostrar("===================================")

        mostrar("\n[5.1] O orquestrador recebeu o consenso:")
        mostrar(self.state.consenso)

        mostrar("\n[5.2] O orquestrador recebeu a crítica:")
        mostrar(critica_recebida)

        mostrar("\n[5.3] Preparando a resposta final...")

        agente = criar_agente(
            role="Orquestrador final",
            goal="Produzir a resposta final para o usuário.",
            backstory=(
                "Você organiza o consenso e a crítica "
                "em uma resposta clara e correta."
            ),
        )

        prompt = f"""
Produza a resposta final para o usuário.

Use o consenso e a crítica.
Corrija os problemas encontrados pelo crítico.

Regras:

- responda diretamente à pergunta;
- não mencione os agentes;
- não invente informações;
- se algo não estiver comprovado no PDF, diga isso;
- escreva em português;
- seja objetivo.

PERGUNTA:
{self.state.pergunta}

CONSENSO:
{self.state.consenso}

CRÍTICA:
{critica_recebida}
"""

        mostrar(
            "\n[5.4] Enviando consenso e crítica "
            "ao orquestrador final..."
        )

        resposta = await chamar_agente(agente, prompt)

        self.state.resposta_final = resposta

        mostrar("\n[5.5] RESPOSTA FINAL PRODUZIDA:")
        mostrar(resposta)

        mostrar("\nFluxo concluído.")

        return resposta


def limpar_caminho(caminho: str) -> str:
    caminho = caminho.strip().strip('"').strip("'")

    if "Caminho do PDF:" in caminho:
        caminho = caminho.rsplit("Caminho do PDF:", 1)[-1].strip()

    return caminho


async def main():
    print("=== CrewAI + Ollama + análise de PDF ===")

    print(
        f"\nDigite o caminho do PDF "
        f"(pressione Enter para usar {PDF_PADRAO}):"
    )

    caminho_digitado = input("> ").strip()

    if caminho_digitado:
        caminho = Path(
            limpar_caminho(caminho_digitado)
        ).expanduser()
    else:
        caminho = Path(PDF_PADRAO)

    if not caminho.is_file():
        mostrar(f"\nArquivo não encontrado: {caminho}")
        mostrar(f"Diretório atual: {Path.cwd()}")
        return

    print("\nDigite sua pergunta:")
    pergunta = input("> ").strip()

    if not pergunta:
        mostrar("A pergunta não pode ficar vazia.")
        return

    mostrar("\nExtraindo texto do PDF...")
    texto_pdf = extrair_texto_pdf(caminho)

    fluxo = FluxoPDF()

    resultado = await fluxo.kickoff_async(
        inputs={
            "texto_pdf": texto_pdf,
            "pergunta": pergunta,
        }
    )

    mostrar("\n==============================")
    mostrar("RESPOSTA FINAL")
    mostrar("==============================")
    mostrar(str(resultado))


if __name__ == "__main__":
    asyncio.run(main())