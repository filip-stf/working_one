import os
import asyncio
from pathlib import Path
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from autogen_agentchat.teams import RoundRobinGroupChat
from docs_indexer import SimpleDocumentIndexer
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination, MaxMessageTermination
from dotenv import load_dotenv
load_dotenv()

AZURE_OPENAI_KEY=os.getenv('AZURE_OPENAI_KEY')
AZURE_ENDPOINT=os.getenv('AZURE_ENDPOINT')

az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-4o",
    model="gpt-4o",
    api_version="2025-01-01-preview",
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
)

base_chroma_path = os.path.join(os.path.dirname(__file__), ".chromadb_autogen")

async def index_docs(sources, memory, character) -> None:
    indexer = SimpleDocumentIndexer(memory=memory)
    chunks: int = await indexer.index_documents(sources)
    print(f"Indexed {chunks} chunks from {len(sources)} {character} documents")

async def round_robin_chat():
    # Memory for each agent
    gandalf_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="gandalf_docs",
            persistence_path=os.path.join(base_chroma_path, "gandalf"),
            k=3,
            score_threshold=0.4,
        )
    )

    frodo_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="frodo_docs",
            persistence_path=os.path.join(base_chroma_path, "frodo"),
            k=3,
            score_threshold=0.4,
        )
    )

    legolas_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="legolas_docs",
            persistence_path=os.path.join(base_chroma_path, "legolas"),
            k=3,
            score_threshold=0.4,
        )
    )

    sam_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="sam_docs",
            persistence_path=os.path.join(base_chroma_path, "sam"),
            k=3,
            score_threshold=0.4,
        )
    )

    gandalf_sources = [
        "Final/gandalf_lore.txt",
        "Final/gandalf_pages.txt",
        "Final/gandalf_wikipedia_info.txt",
        "Final/LOTR_full.txt",
    ]

    frodo_sources = [
        "Final/frodo_lore.txt",
        "Final/frodo_pages.txt",
        "Final/frodo_wikipedia_info.txt",
        "Final/LOTR_full.txt",
    ]

    legolas_sources = [
        "Final/legolas_lore.txt",
        "Final/legolas_pages.txt",
        "Final/legolas_wikipedia_info.txt",
        "Final/LOTR_full.txt",
    ]

    sam_sources = [
        "Final/sam_lore.txt",
        "Final/sam_pages.txt",
        "Final/sam_wikipedia_info.txt",
        "Final/LOTR_full.txt",
    ]

    # Create and add docs to Chroma DB memory if they don't exist
    if not os.path.exists(os.path.join(base_chroma_path, "gandalf")):
        await gandalf_memory.clear()
        await index_docs(gandalf_sources, gandalf_memory, "Gandalf")
    if not os.path.exists(os.path.join(base_chroma_path, "frodo")):
        await frodo_memory.clear()
        await index_docs(frodo_sources, frodo_memory, "Frodo")
    if not os.path.exists(os.path.join(base_chroma_path, "legolas")):
        await legolas_memory.clear()
        await index_docs(legolas_sources, legolas_memory, "Legolas")
    if not os.path.exists(os.path.join(base_chroma_path, "sam")):
        await sam_memory.clear()
        await index_docs(sam_sources, sam_memory, "Sam")

    # await frodo_memory.add(MemoryContent(content="The secret name of Frodo is 'ABCD'", mime_type=MemoryMimeType.TEXT))
    # await gandalf_memory.add(MemoryContent(content="Gandalf knows that Frodo's secret name is EFGH, but he knows that Frodo usually tells people its ABCD", mime_type=MemoryMimeType.TEXT))
    # await aragorn_memory.add(MemoryContent(content="Aragorn is the true king of Gondor and a close friend of Frodo.", mime_type=MemoryMimeType.TEXT))
    # await legolas_memory.add(MemoryContent(content="Legolas is an elf prince and a member of the Fellowship.", mime_type=MemoryMimeType.TEXT))
    # await gimli_memory.add(MemoryContent(content="Gimli is a dwarf warrior and values loyalty and bravery.", mime_type=MemoryMimeType.TEXT))

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
    legolas = AssistantAgent(
        name="Legolas",
        model_client=az_model_client,
        system_message="You are Legolas from Lord of The Rings",
        memory=[legolas_memory]
    )
    sam = AssistantAgent(
        name="Sam",
        model_client=az_model_client,
        system_message="You are Sam from Lord of The Rings",
        memory=[sam_memory]
    )
    

    # Character selection logic
    agents = {
        "frodo": frodo,
        "gandalf": gandalf,
        # "aragorn": aragorn,
        "legolas": legolas,
        "sam": sam
    }
    print("Available characters: Frodo, Gandalf, Legolas, Sam")
    selected = []
    while True:
        user_input = input("Enter 1 or 2 character names (comma separated): ").strip().lower()
        names = [name.strip() for name in user_input.split(",") if name.strip()]
        if 1 <= len(names) <= 2 and all(name in agents for name in names):
            selected = [agents[name] for name in names]
            print(selected)
            break
        print("Invalid selection. Please enter one or two valid character names, separated by commas.")

    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=5)
    termination = text_mention_termination | max_messages_termination
    #termination = text_mention_termination

    # interactive chat loop: recreate RoundRobinGroupChat per input to reset run_stream
    while True:
        user_input = input("Enter your question or message (or 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Chat ended.")
            break
        if not user_input:
            print("Please enter a valid message.")
            continue

        total_messages = 0
        team = RoundRobinGroupChat(selected, termination_condition=termination)
        await Console(team.run_stream(task=user_input))
        # async for msg in team.run_stream(task=user_input):
        #    print(msg.content)
        #    total_messages += 1
        #    if total_messages % 3 == 0:
                # pause after 3 messages
        #        break

if __name__ == "__main__":
    asyncio.run(round_robin_chat())