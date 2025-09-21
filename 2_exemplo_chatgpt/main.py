import os
import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA


# === 1) Ler CSV e converter em Document ===
csv_path = "clientes.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"O arquivo {csv_path} não foi encontrado.")

df = pd.read_csv(csv_path)
print(df)

docs = []
for idx, row in df.iterrows():
    # Cria um texto concatenando todas as colunas
    content = ", ".join(f"{col}: {row[col]}" for col in df.columns)
    docs.append(Document(page_content=content, metadata={"linha": idx}))


# === 2) Criar embeddings e Chroma DB ===
embeddings = OllamaEmbeddings(model= "mxbai-embed-large")

db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"  # persistência automática
)

retriever = db.as_retriever(search_kwargs={"k": 3})


# === 3) Configurar LLM e prompt ===
llm = OllamaLLM(model="llama3.2")

prompt = ChatPromptTemplate.from_template(
    "Você é um assistente que responde de forma clara. "
    "Use os dados fornecidos para responder a pergunta.\n\n"
    "Contexto:\n{context}\n\nPergunta: {question}"
)


# === 4) Criar chain de QA ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


# === 5) Fazer uma pergunta ===
query = "Quais clientes têm o maior ticket médio?"
result = qa_chain.invoke({"query": query})

print("Resposta:", result["result"])
print("\nFontes usadas:")
for doc in result["source_documents"]:
    print("-", doc.metadata)
