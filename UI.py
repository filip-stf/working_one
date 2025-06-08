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

# --- MOVE CHAT INPUT TO TOP OF PAGE ---
user_msg = None
user_input_pending = False
# Define callback to queue and clear user input
def queue_message():
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.messages.append(("You", user_input))
        st.session_state._pending_user_msg = user_input
        st.session_state.user_input = ""

# Chat interaction: use text_input with on_change callback
if "selected_characters" in st.session_state and st.session_state.selected_characters:
    # Only text input at bottom
    st.text_input("Type your message here…", key="user_input", placeholder="Type your message here…", on_change=queue_message)
# --- END CHAT INPUT AT TOP ---

st.title("Lord of The Rings RAG Chat")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_characters" not in st.session_state:
    st.session_state.selected_characters = None

# Manage multiple chat sessions
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
    st.session_state.current_session = None

# Sidebar: session selection and creation
with st.sidebar:
    st.header("Chat Sessions")
    # Button to create a new chat session
    if st.button("New Chat", key="new_chat_btn"):
        new_name = f"Chat {len(st.session_state.chat_sessions) + 1}"
        st.session_state.chat_sessions[new_name] = {"messages": [], "selected_characters": None}
        st.session_state.current_session = new_name
        # reset main session state
        st.session_state.messages = []
        st.session_state.selected_characters = None
        st.rerun()
    # Dropdown to select existing sessions
    session_names = list(st.session_state.chat_sessions.keys())
    if session_names:
        default_index = session_names.index(st.session_state.current_session) if st.session_state.current_session in session_names else 0
        selection = st.selectbox("Select session", session_names, index=default_index, key="session_select")
        if st.session_state.current_session != selection:
            st.session_state.current_session = selection
            data = st.session_state.chat_sessions.get(selection, {})
            st.session_state.messages = data.get("messages", [])
            st.session_state.selected_characters = data.get("selected_characters", None)
    # Reset current session
    if st.session_state.current_session:
        if st.button("Reset Chat", key="reset_session"):
            st.session_state.chat_sessions[st.session_state.current_session]["messages"] = []
            st.session_state.chat_sessions[st.session_state.current_session]["selected_characters"] = None
            st.session_state.messages = []
            st.session_state.selected_characters = None
            st.rerun()

# Prepare base64-encoded icons for inline embedding
import base64 as _base64
_icons = {name: f"icons/{name}.png" for name in ["frodo","gandalf","legolas","sam"]}
_icon_b64 = {}
for _name, _path in _icons.items():
    try:
        with open(_path, 'rb') as _f:
            _icon_b64[_name] = _base64.b64encode(_f.read()).decode()
    except FileNotFoundError:
        _icon_b64[_name] = None

# Character selection
if st.session_state.selected_characters is None:
    st.markdown("<div style='text-align:center; margin:20px 0;'><h3>Select Characters</h3></div>", unsafe_allow_html=True)
    # Ściśnij odstępy tekstu checkboxa i etykiet
    st.markdown(
        """
        <style>
        .stCheckbox {margin: 0 !important; padding: 0 !important;}
        .stCheckbox label {margin: 0 !important; padding: 0 !important; line-height:1 !important;}
        </style>
        """,
        unsafe_allow_html=True
    )
    # Dodajemy niewidoczny spacer na początku i na końcu listy
    cols = st.columns(8, gap="medium")
    chars = ["spacer_start_1", "spacer_start_2", "frodo", "gandalf", "legolas", "sam", "spacer_end_1", "spacer_end_2"]
    labels = {
        "frodo":"Frodo", "gandalf":"Gandalf", "legolas":"Legolas", "sam":"Sam",
        "spacer_start_1":"", "spacer_end_1":"", "spacer_start_2":"", "spacer_end_2":""
    }
    selected = []
    for col, name in zip(cols, chars):
        with col:
            if name.startswith("spacer_"):
                # invisible spacer at beginning or end
                st.markdown(
                    "<div style='visibility:hidden; width:80px; height:60px;'></div>",
                    unsafe_allow_html=True
                )
                continue
            checked = st.checkbox(labels[name], key=f"select_{name}")
            # base64 inline icon
            img_src = (f"data:image/png;base64,{_icon_b64[name]}" if _icon_b64.get(name) else f"icons/{name}.png")
            border = "3px solid #2c3e50" if checked else "1px solid transparent"
            # icon above checkbox label
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<img src='{img_src}' width='80' style='border:{border};border-radius:8px;margin-bottom:4px;'/>"
                f"</div>", unsafe_allow_html=True
            )
            if checked:
                selected.append(name)
    # Start chat button
    if len(selected) >= 1 and len(selected) <= 2:
        if st.button("Start Chat"):
            # Create a new chat session, same as clicking 'New Chat'
            new_name = f"Chat {len(st.session_state.chat_sessions) + 1}"
            st.session_state.chat_sessions[new_name] = {
                "messages": [],
                "selected_characters": selected
            }
            st.session_state.current_session = new_name
            st.session_state.messages = []
            st.session_state.selected_characters = selected
            st.rerun()
    else:
        st.button("Start Chat", disabled=True)
else:
    st.write(f"Chatting with: {', '.join(st.session_state.selected_characters).title()}")

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

# Add background image styling (after input box)
import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    bg_img = get_base64_of_bin_file("background.png")
    st.markdown(
        f"""
        <style>
        html, body, .stApp {{
            height: 100vh !important;
            min-height: 100vh !important;
            width: 100vw !important;
            min-width: 100vw !important;
            margin: 0 !important;
            padding: 0 !important;
            background-image: url('data:image/png;base64,{bg_img}') !important;
            background-size: cover !important;
            background-position: center center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        }}
        .stApp {{
            background: transparent !important;
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
        /* Make main container full height with basic spacing */
        .main .block-container {{
            min-height: 100vh !important;
            padding: 0.5rem 2rem 2rem 2rem !important;
            margin: 0 !important;
            background-color: transparent !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            box-sizing: border-box !important;
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
        /* Force chat message text to be black */
        .stChatMessage p, .stChatMessage div, .stChatMessage span {{
            color: #000000 !important;
        }}
        /* Float text input box at bottom of screen */
        div[data-testid="stTextInput"] {{
            position: fixed !important;
            bottom: 20px !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            width: 100% !important;
            max-width: 800px !important;
            z-index: 1000 !important;
        }}
        /* Make the input field background white */
        div[data-testid="stTextInput"] input {{
            background-color: #ffffff !important;
            color: #000000 !important;
        }}
        /* Sidebar translucent background */
        aside[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] {{
            background-color: rgba(255, 255, 255, 0.5) !important;
            backdrop-filter: blur(5px) !important;
            border-radius: 10px !important;
            padding: 1rem !important;
        }}
        /* Character icon selection wrapper styling */
        #char-selection-wrapper {{
            position: relative !important;
            height: auto !important;
            margin: 20px auto 40px !important;
            width: 60% !important;
            background-color: rgba(255,255,255,0.8) !important;
            border: 2px solid rgba(0,0,0,0.2) !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
            padding: 20px 0 10px 0 !important;
            z-index: 1 !important;
        }}
        /* Character icon selection spacing */
        #char-selection-wrapper > div[data-testid="column"] {{
            padding: 0 !important;
            margin: 0 !important;
            position: static !important;
            z-index: auto !important;
        }}
        #char-selection img {{
            margin: 0 auto 2px !important;
            display: block !important;
        }}
        #char-selection .stCheckbox {{
            margin: 0 !important;
            padding: 0 !important;
            text-align: center !important;
        }}
        /* Reduce space above checkbox labels */
        #char-selection label {{
            margin-top: 2px !important;
            margin-bottom: 0 !important;
        }}
        #char-selection .stCheckbox > label {{
            margin: 0 !important;
            text-align: center !important;
            display: block !important;
            justify-content: center !important;
        }}
        #char-selection .stCheckbox input[type="checkbox"] {{
            margin: 0 auto !important;
        }}
        /* Remove Streamlit's default spacing above checkboxes */
        #char-selection .stCheckbox > div {{
            margin-top: 0 !important;
            padding-top: 0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.warning("Background image not found. Using default background.")

# Icon mapping for characters
icon_paths = {
    "frodo": "icons/frodo.png",
    "gandalf": "icons/gandalf.png",
    "legolas": "icons/placeholder.png",
    "sam": "icons/placeholder.png",
    "You": "icons/placeholder.png"  # User icon
}

# After run_chat is defined and before displaying chat history:
if hasattr(st.session_state, '_pending_user_msg') and st.session_state._pending_user_msg:
    try:
        loop = asyncio.new_event_loop()
        replies = loop.run_until_complete(run_chat(st.session_state._pending_user_msg, st.session_state.selected_characters))
        loop.close()
        for speaker, text in replies:
            st.session_state.messages.append((speaker, text))
    except Exception as e:
        st.error(f"Chat error: {e}")
    st.session_state._pending_user_msg = None

# Persist current session data after processing messages
if st.session_state.current_session:
    st.session_state.chat_sessions[st.session_state.current_session] = {
        "messages": st.session_state.messages,
        "selected_characters": st.session_state.selected_characters
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