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
        async for msg in team.run_stream(task=user_input):
            print(msg.content)
            total_messages += 1
            if total_messages % 3 == 0:
                # pause after 3 messages
                break

# Dispatch to Streamlit when run via `python rag_memory.py`
if __name__ == "__main__":
    import sys
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", sys.argv[0]] + sys.argv[1:]
    sys.exit(stcli.main())

# Streamlit UI (only when invoked via `streamlit run`)
import streamlit as st

st.set_page_config(page_title="LOTR Chat", layout="centered")
st.title("Lord of The Rings Chat")

colors = {
    "Frodo": "#F4A460",
    "Gandalf": "#D3D3D3",
    "Aragorn": "#98FB98",
    "Legolas": "#90EE90",
    "Gimli": "#FFD700"
}

cols = st.columns(len(colors))
for col, name in zip(cols, colors):
    with col:
        st.markdown(
            f"<div style='width:100px; height:100px; background-color: {colors[name]};'></div>",
            unsafe_allow_html=True
        )
        st.write(name)

st.info("Character selection and chat functionality coming soon.")