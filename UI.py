import streamlit as st

st.set_page_config(page_title="LOTR Chat", layout="wide")

import asyncio
import os
import sys
from io import StringIO
from contextlib import redirect_stdout
from unittest.mock import patch

st.title("Lord of The Rings Group Chat")

# initialize chat state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_characters" not in st.session_state:
    st.session_state.selected_characters = None

# Character selection
if st.session_state.selected_characters is None:
    st.subheader("Select Characters")
    st.write("Available characters: Frodo, Gandalf, Aragorn, Legolas, Gimli")
    
    character_input = st.text_input("Enter 1 or 2 character names (comma separated):")
    if st.button("Start Chat") and character_input:
        names = [name.strip().lower() for name in character_input.split(",") if name.strip()]
        valid_names = ["frodo", "gandalf", "aragorn", "legolas", "gimli"]
        if 1 <= len(names) <= 2 and all(name in valid_names for name in names):
            st.session_state.selected_characters = names
            st.rerun()
        else:
            st.error("Invalid selection. Please enter one or two valid character names, separated by commas.")
else:
    st.write(f"Chatting with: {', '.join(st.session_state.selected_characters).title()}")
    
    if st.button("Reset Characters"):
        st.session_state.selected_characters = None
        st.session_state.messages = []
        st.rerun()

# chat functionality (only if characters selected)
if st.session_state.selected_characters:
    user_msg = st.chat_input("Your message")
    if user_msg:
        st.session_state.messages.append(("You", user_msg))
        
        # Capture output from rag_memory logic
        try:
            # Create agents and run chat directly
            from autogen_agentchat.agents import AssistantAgent
            from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
            from autogen_agentchat.conditions import TextMentionTermination
            from autogen_agentchat.teams import RoundRobinGroupChat
            from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
            
            # Create Azure client
            az_model_client = AzureOpenAIChatCompletionClient(
                azure_deployment="gpt-4o",
                model="gpt-4o-2024-11-20",
                api_version="2024-12-01-preview",
                azure_endpoint=os.getenv('AZURE_ENDPOINT'),
                api_key=os.getenv('AZURE_OPENAI_KEY'),
            )
            
            async def run_chat():
                # Create agents with memories
                memories = {
                    "frodo": "The secret name of Frodo is 'ABCD'",
                    "gandalf": "Gandalf knows that Frodo's secret name is EFGH, but he knows that Frodo usually tells people its ABCD",
                    "aragorn": "Aragorn is the true king of Gondor and a close friend of Frodo.",
                    "legolas": "Legolas is an elf prince and a member of the Fellowship.",
                    "gimli": "Gimli is a dwarf warrior and values loyalty and bravery."
                }
                
                agents = {}
                for name, content in memories.items():
                    mem = ListMemory()
                    await mem.add(MemoryContent(content=content, mime_type=MemoryMimeType.TEXT))
                    agents[name] = AssistantAgent(
                        name=name.title(),
                        model_client=az_model_client,
                        system_message=f"You are {name.title()} from Lord of The Rings",
                        memory=[mem]
                    )
                
                # Select agents based on user choice
                selected = [agents[name] for name in st.session_state.selected_characters]
                
                # Run chat
                term = TextMentionTermination("TERMINATE")
                team = RoundRobinGroupChat(selected, termination_condition=term)
                
                responses = []
                async for msg in team.run_stream(task=user_msg):
                    # msg.source should contain the agent name
                    speaker = getattr(msg, 'source', 'Agent')
                    content = getattr(msg, 'content', str(msg))
                    responses.append((speaker, content))
                    if len(responses) >= 3:  # Limit to 3 responses
                        break
                
                return responses
            
            replies = asyncio.run(run_chat())
            for speaker, content in replies:
                st.session_state.messages.append((speaker, content))
                     
        except Exception as e:
            st.error(f"Error running chat: {e}")

# render chat history
for speaker, text in st.session_state.messages:
    with st.chat_message(speaker):
        st.write(text)
