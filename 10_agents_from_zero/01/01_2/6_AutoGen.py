import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient


async def main():
    modelo = OllamaChatCompletionClient(
        model="qwen2.5-coder:7b"
    )

    agente = AssistantAgent(
        name="assistente",
        model_client=modelo,
    )

    await Console(
        agente.run_stream(
            task="O que é um agente de IA?"
        )
    )

    await modelo.close()


asyncio.run(main())