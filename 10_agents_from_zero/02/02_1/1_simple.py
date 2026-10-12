from pypdf import PdfReader
import ollama


MODEL = "qwen2.5-coder:7b"
PDF_PATH = "Modelo-de-contrato-de-aluguel-1.pdf"


def ler_pdf(caminho):
    leitor = PdfReader(caminho)

    texto = ""

    for pagina in leitor.pages:
        texto += pagina.extract_text() or ""

    return texto


def perguntar_ao_ollama(texto_pdf):
    prompt = f"""
Você é um agente que lê documentos.

Use somente o conteúdo do PDF para responder.

CONTEÚDO DO PDF:
{texto_pdf}

Pergunta:
Qual é o assunto principal deste documento?
"""

    resposta = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "num_predict": 512,
        },
    )

    return resposta.message.content


def main():
    texto_pdf = ler_pdf(PDF_PATH)

    if not texto_pdf.strip():
        raise ValueError(
            "Não foi possível extrair texto do PDF."
        )

    resposta = perguntar_ao_ollama(texto_pdf)

    print(resposta)


if __name__ == "__main__":
    main()