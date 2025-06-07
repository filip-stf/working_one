import streamlit as st
import os
import sys
import asyncio
from pathlib import Path

# Ensure workspace root is in path to import rag_memory
sys.path.append(str(Path(__file__).resolve().parents[0]))

from dotenv import load_dotenv
load_dotenv()

import rag_memory
from rag_memory import index_docs, base_chroma_path
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.memory import MemoryContent

# Streamlit UI configuration
st.set_page_config(page_title="LOTR RAG Chat", layout="wide")

# Add background image styling
import base64

# Function to encode image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Simple background with working chat input
try:
    bg_img = get_base64_of_bin_file("background.png")
    st.markdown(
        f"""
        <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Tangerine:wght@400;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Lugrasimo&family=Tangerine:wght@400;700&display=swap');
          /* Apply Font Family to all elements */
        html, body, .stApp, p, div, h4, h5, h6, button, input, textarea, .stMarkdown, .stText {{
            font-family: "Lugrasimo", cursive;
            font-size: 20px !important;
        }}
        
        /* Special styling for headers */
        h1 {{
            font-family: "Tangerine", cursive !important;
            font-size: 60px !important;
            font-weight: 700 !important;
        }}
        
        h2, h3 {{
            font-family: "Tangerine", cursive !important;
            font-size: 42px !important;
            font-weight: 700 !important;
        }}
        
        /* Remove default padding and margins */
        .main .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: none !important;
        }}
        
        /* Ensure full page coverage without layout shifts */
        html, body {{
            height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow-x: hidden !important;
        }}
        
        /* Set background on the root app container to cover full page */
        .stApp {{
            background-image: url("data:image/png;base64,{bg_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            min-height: 100vh !important;
            width: 100vw !important;
            margin: 0 !important;
            padding: 0 !important;
            position: relative !important;
        }}
        
        /* Ensure all main content stays above background and fills page */
        .main {{
            position: relative;
            z-index: 10;
            min-height: 100vh !important;
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        
        /* Remove default streamlit spacing */
        .stApp > header {{
            display: none !important;
        }}
        
        /* Make main container full height with proper spacing for input */
        .main .block-container {{
            min-height: 100vh !important;
            padding: 0.5rem 2rem 180px 2rem !important;
            margin: 0 !important;
            background-color: transparent !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            box-sizing: border-box !important;
        }}
        
        /* Chat input window - positioned higher over background */
        .stChatInput {{
            position: fixed !important;
            bottom: 20px !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            width: calc(100% - 40px) !important;
            max-width: 800px !important;
            z-index: 1000 !important;
        }}
        
        /* Chat input container styling - white mode with smaller height */
        .stChatInput > div {{
            background-color: #ffffff !important;
            border-radius: 20px !important;
            padding: 12px 16px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
            border: 1px solid #e0e0e0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }}
        
        /* Additional targeting for input container */
        div[data-testid="stChatInput"] > div,
        .stChatInput .stChatInputContainer {{
            background-color: #ffffff !important;
            border-radius: 20px !important;
            padding: 12px 16px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
            border: 1px solid #e0e0e0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }}
          /* Input field styling - clean white with smaller height */
        .stChatInput input,
        .stChatInput textarea,
        div[data-testid="stChatInput"] input,
        div[data-testid="stChatInput"] textarea,
        .stChatInput div[contenteditable="true"],
        div[data-testid="stChatInput"] div[contenteditable="true"] {{
            background-color: #ffffff !important;
            background: #ffffff !important;
            border: 1px solid #d0d0d0 !important;
            border-radius: 15px !important;
            padding: 10px 16px !important;
            font-size: 24px !important;
            color: #333333 !important;
            width: 100% !important;
            box-sizing: border-box !important;
            height: 44px !important;
            min-height: 44px !important;
            max-height: 44px !important;
        }}
        
        /* Remove any default Streamlit input styling that might cause black sections */
        .stChatInput * {{
            background-color: transparent !important;
        }}
        
        .stChatInput input,
        .stChatInput textarea,
        div[data-testid="stChatInput"] input,
        div[data-testid="stChatInput"] textarea {{
            background-color: #ffffff !important;
            background: #ffffff !important;
        }}
        
        /* Send button styling - center it */
        .stChatInput button,
        div[data-testid="stChatInput"] button {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: auto !important;
            background-color: #4A90E2 !important;
            border: none !important;
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
            color: white !important;
        }}
        
        /* Input field focus state - blue accent */
        .stChatInput input:focus,
        .stChatInput textarea:focus,
        div[data-testid="stChatInput"] input:focus,
        div[data-testid="stChatInput"] textarea:focus {{
            outline: none !important;
            border-color: #4A90E2 !important;
            box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.2) !important;
            background-color: #ffffff !important;
            background: #ffffff !important;
        }}
        
        h1 {{
            color: #2c3e50;
            text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.9);
            text-align: center;
            margin-bottom: 10px;
            margin-top: 0;
            padding-top: 0;
        }}
        
        /* Chat messages container styling */
        .stChatMessage {{
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-radius: 15px !important;
            margin-bottom: 10px !important;
            padding: 10px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
            color: #000000 !important;
        }}
        
        /* Ensure chat messages area doesn't overlap with input */
        .main .block-container > div {{
            max-height: calc(100vh - 140px) !important;
            overflow-y: auto !important;
            padding-bottom: 20px !important;
        }}          /* Force chat message text to be black with differentiated sizing */
        .stChatMessage div, .stChatMessage span {{
            color: #000000 !important;
            font-size: 30px !important;
        }}
        
        /* Make paragraphs inside chat messages smaller */
        .stChatMessage p {{
            color: #000000 !important;
            font-size: 20px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.warning("Background image not found. Using default background.")

st.title("Lord of The Rings Chat")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_characters" not in st.session_state:
    st.session_state.selected_characters = None

# Character selection
if st.session_state.selected_characters is None:
    st.write("Select Characters")
    st.write("Available characters: Frodo, Gandalf, Legolas, Sam")
    chars = st.text_input("Enter 1 or 2 character names (comma separated):")
    if st.button("Start Chat") and chars:
        names = [n.strip().lower() for n in chars.split(",") if n.strip()]
        valid = ["frodo", "gandalf", "legolas", "sam"]
        if 1 <= len(names) <= 2 and all(n in valid for n in names):
            st.session_state.selected_characters = names
            st.rerun()
        else:
            st.error("Invalid selection. Please enter one or two valid character names.")
else:
    st.write(f"Chatting with: {', '.join(st.session_state.selected_characters).title()}")
    if st.button("Reset"):  # reset selection
        st.session_state.selected_characters = None
        st.session_state.messages = []
        st.rerun()

# Helper to prepare and index memories
async def prepare_memories():
    configs = {
        "frodo": ["Final/frodo_lore.txt", "Final/frodo_pages.txt", "Final/frodo_wikipedia_info.txt", "Final/LOTR_full.txt"],
        "gandalf": ["Final/gandalf_lore.txt", "Final/gandalf_pages.txt", "Final/gandalf_wikipedia_info.txt", "Final/LOTR_full.txt"],
        "legolas": ["Final/legolas_lore.txt", "Final/legolas_pages.txt", "Final/legolas_wikipedia_info.txt", "Final/LOTR_full.txt"],
        "sam": ["Final/sam_lore.txt", "Final/sam_pages.txt", "Final/sam_wikipedia_info.txt", "Final/LOTR_full.txt"],
    }
    memories = {}
    for name, sources in configs.items():
        path = os.path.join(base_chroma_path, name)
        mem = ChromaDBVectorMemory(
            config=PersistentChromaDBVectorMemoryConfig(
                collection_name=f"{name}_docs",
                persistence_path=path,
                k=3,
                score_threshold=0.4
            )
        )
        if not os.path.exists(path):
            await mem.clear()
            await index_docs(sources, mem, name.title())
        memories[name] = mem
    return memories

# Run chat with selected agents
async def run_chat(user_input, selected_names):
    # Prepare vector memories
    memories = await prepare_memories()
    # Create Azure client
    client = AzureOpenAIChatCompletionClient(
        azure_deployment="gpt-4o",
        model="gpt-4o",
        api_version="2025-01-01-preview",
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
    )
    # Instantiate agents
    agents = []
    for name in selected_names:
        mem = memories[name]
        agent = AssistantAgent(
            name=name.title(),
            model_client=client,
            system_message=f"You are {name.title()} from Lord of The Rings",
            memory=[mem]
        )
        agents.append(agent)
    # Setup termination condition
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=5)
    team = RoundRobinGroupChat(agents, termination_condition=termination)
    # Run and collect responses
    responses = []
    async for msg in team.run_stream(task=user_input):
        speaker = getattr(msg, 'source', '').lower()
        # only show messages from chosen characters
        if speaker not in st.session_state.selected_characters:
            continue
        raw = getattr(msg, 'content', msg)
        # skip memory retrieval artifacts or lists
        if isinstance(raw, MemoryContent) or isinstance(raw, list):
            continue
        # use actual text
        content = str(raw)
        responses.append((speaker, content))
        if len(responses) >= 3:
            break
    return responses

# Chat interaction
if st.session_state.selected_characters:
    prompt = st.chat_input("Your message")
    if prompt:
        st.session_state.messages.append(("You", prompt))
        try:
            # run coroutine on a new event loop to avoid asyncio.run shutdown issues
            loop = asyncio.new_event_loop()
            replies = loop.run_until_complete(run_chat(prompt, st.session_state.selected_characters))
            loop.close()
            for speaker, text in replies:
                st.session_state.messages.append((speaker, text))
        except Exception as e:
            st.error(f"Chat error: {e}")

# Icon mapping for characters
icon_paths = {
    "frodo": "icons/frodo.png",
    "gandalf": "icons/gandalf.png",
    "legolas": "icons/placeholder.png",
    "sam": "icons/placeholder.png",
    "You": "icons/placeholder.png"  # User icon
}

# Display chat history with proper alignment
for speaker, message in st.session_state.messages:
    if speaker == "You":
        # Right-align user messages with icon on the left of the message
        col1, col2, col3 = st.columns([1, 3, 1])
        with col3:
            # Create a container for right-aligned content
            with st.container():
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: 10px;">
                        <div style="background-color: #ffffff; color: black; padding: 10px 15px; border-radius: 18px; max-width: 80%; text-align: right; margin-right: 10px; border: 1px solid #ddd;">
                            {message}
                        </div>
                        <img src="icons/placeholder.png" style="width: 40px; height: 40px; border-radius: 50%;" />
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        # Left-align character messages (default)
        icon = icon_paths.get(speaker.lower(), "🧙‍♂️")
        if icon.endswith('.png'):
            # For image icons, use standard chat message
            with st.chat_message(speaker, avatar=icon):
                st.write(message)
        else:
            # For emoji icons
            with st.chat_message(speaker, avatar=icon):
                st.write(message)
