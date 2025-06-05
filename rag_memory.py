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
    await frodo_memory.add(MemoryContent(content="The secret name of Frodo is 'ABCD'", mime_type=MemoryMimeType.TEXT))
    await gandalf_memory.add(MemoryContent(content="Gandalf knows that Frodo's secret name is EFGH, but he knows that Frodo usually tells people its ABCD", mime_type=MemoryMimeType.TEXT))

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
    from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination, MaxMessageTermination

    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=3)
    termination = text_mention_termination | max_messages_termination

    team = RoundRobinGroupChat([gandalf, frodo], termination_condition=termination)
    result = await Console(team.run_stream(task="What is Frodo's secret name?"))
    print(result)

if __name__ == "__main__":
    asyncio.run(round_robin_chat())