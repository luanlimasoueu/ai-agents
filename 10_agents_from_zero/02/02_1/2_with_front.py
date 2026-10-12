import io

import ollama
import streamlit as st
from pypdf import PdfReader


MODEL = "qwen2.5-coder:7b"


st.set_page_config(
    page_title="Agente de PDF",
    page_icon="📄",
)

st.title("📄 Agente leitor de PDF")

arquivo = st.file_uploader(
    "Escolha um arquivo PDF",
    type=["pdf"],
)

if arquivo:
    leitor = PdfReader(io.BytesIO(arquivo.getvalue()))

    texto_pdf = ""

    for pagina in leitor.pages:
        texto_pdf += pagina.extract_text() or ""

    if not texto_pdf.strip():
        st.error("Não foi possível extrair texto deste PDF.")
        st.stop()

    st.success(
        f"PDF carregado com {len(leitor.pages)} página(s)."
    )

    pergunta = st.text_input(
        "Faça uma pergunta sobre o PDF:"
    )

    if st.button("Perguntar"):
        if not pergunta.strip():
            st.warning("Digite uma pergunta.")
        else:
            prompt = f"""
Você é um agente leitor de documentos.

Responda usando somente o conteúdo do PDF.

CONTEÚDO DO PDF:
{texto_pdf}

PERGUNTA:
{pergunta}
"""

            with st.spinner("O agente está lendo o PDF..."):
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

            st.subheader("Resposta")
            st.write(resposta.message.content)