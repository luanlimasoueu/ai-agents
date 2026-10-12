from crewai import Agent, Task, Crew, LLM


modelo = LLM(
    model="ollama/qwen2.5-coder:7b",
    base_url="http://127.0.0.1:11434",
)


agente = Agent(
    role="Assistente",
    goal="Responder perguntas de forma simples",
    backstory="Você é um assistente objetivo.",
    llm=modelo,
)


tarefa = Task(
    description="Responda: o que é um agente de IA?",
    expected_output="Uma explicação curta.",
    agent=agente,
)


crew = Crew(
    agents=[agente],
    tasks=[tarefa],
)


print(crew.kickoff())