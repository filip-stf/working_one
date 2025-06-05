import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat

AZURE_OPENAI_KEY=os.getenv('AZURE_OPENAI_KEY')
AZURE_ENDPOINT=os.getenv('AZURE_ENDPOINT')

az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-4o",
    model="gpt-4o",
    api_version="2024-12-01-preview",
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
)


async def round_robin_chat():
    # Memory for each agent
    frodo_memory = ListMemory()
    gandalf_memory = ListMemory()
    aragorn_memory = ListMemory()
    legolas_memory = ListMemory()
    gimli_memory = ListMemory()
    await frodo_memory.add(MemoryContent(content="The secret name of Frodo is 'ABCD'", mime_type=MemoryMimeType.TEXT))
    await gandalf_memory.add(MemoryContent(content="Gandalf knows that Frodo's secret name is EFGH, but he knows that Frodo usually tells people its ABCD", mime_type=MemoryMimeType.TEXT))
    await aragorn_memory.add(MemoryContent(content="Aragorn is the true king of Gondor and a close friend of Frodo.", mime_type=MemoryMimeType.TEXT))
    await legolas_memory.add(MemoryContent(content="Legolas is an elf prince and a member of the Fellowship.", mime_type=MemoryMimeType.TEXT))
    await gimli_memory.add(MemoryContent(content="Gimli is a dwarf warrior and values loyalty and bravery.", mime_type=MemoryMimeType.TEXT))

    frodo = AssistantAgent(
        name="Frodo",
        model_client=az_model_client,
        system_message="You are Frodo from Lord of The Rings",
        memory=[frodo_memory]
    )
    gandalf = AssistantAgent(
        name="Gandalf",
        model_client=az_model_client,
        system_message="You are Gandalf from Lord of The Rings",
        memory=[gandalf_memory]
    )
    aragorn = AssistantAgent(
        name="Aragorn",
        model_client=az_model_client,
        system_message="You are Aragorn from Lord of The Rings",
        memory=[aragorn_memory]
    )
    legolas = AssistantAgent(
        name="Legolas",
        model_client=az_model_client,
        system_message="You are Legolas from Lord of The Rings",
        memory=[legolas_memory]
    )
    gimli = AssistantAgent(
        name="Gimli",
        model_client=az_model_client,
        system_message="You are Gimli from Lord of The Rings",
        memory=[gimli_memory]
    )
    from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination

    # Character selection logic
    agents = {
        "frodo": frodo,
        "gandalf": gandalf,
        "aragorn": aragorn,
        "legolas": legolas,
        "gimli": gimli
    }
    print("Available characters: Frodo, Gandalf, Aragorn, Legolas, Gimli")
    selected = []
    while True:
        user_input = input("Enter 1 or 2 character names (comma separated): ").strip().lower()
        names = [name.strip() for name in user_input.split(",") if name.strip()]
        if 1 <= len(names) <= 2 and all(name in agents for name in names):
            selected = [agents[name] for name in names]
            break
        print("Invalid selection. Please enter one or two valid character names, separated by commas.")

    text_mention_termination = TextMentionTermination("TERMINATE")
    termination = text_mention_termination

    team = RoundRobinGroupChat(selected, termination_condition=termination)
    result = await Console(team.run_stream(task="What is Frodo's secret name?"))
    print(result)

if __name__ == "__main__":
    asyncio.run(round_robin_chat())