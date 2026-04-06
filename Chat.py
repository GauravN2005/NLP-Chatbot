import streamlit as st
import io # Used for in-memory file for download button
from google import genai
from google.genai import types
from google.genai.errors import APIError
import os
from dotenv import load_dotenv

# --- CONFIGURATION AND INITIALIZATION ---

# 1. API Key retrieval from st.secrets (Required for Streamlit Cloud deployment)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    # Local fallback to .env when secrets are not configured or secrets.toml missing
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error(" GEMINI_API_KEY not found! Add it to Streamlit Secrets or define it in a local .env file.")
        st.stop()

# Cache the Gemini client to prevent recreation on each rerun
@st.cache_resource
def get_gemini_client():
    """Initialize and cache the Gemini client."""
    return genai.Client(api_key=GEMINI_API_KEY)

# Initialize cached Gemini client
client = get_gemini_client()

# --- SESSION STATE MANAGEMENT ---

# Initialize chat history (used for UI display)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize system instruction in session state
if "system_instruction" not in st.session_state:
    SYSTEM_INSTRUCTION_DEFAULT = (
        "You are an engaging and helpful AI StudyMate specializing in AI, ML, and Data Science. "
        "Your goal is to provide accurate, easy-to-understand, and structured explanations. "
        "Use Markdown, lists, and code blocks liberally to format your answers. "
        "Only answer questions related to studying AI/ML/Data Science topics. If a request is out of scope, "
        "politely ask the user to rephrase it to a study-related topic."
    )
    st.session_state["system_instruction"] = SYSTEM_INSTRUCTION_DEFAULT

# --- CORE FUNCTIONALITY ---

# Basic study-topic relevance checker
STUDY_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "ml", "deep learning",
    "neural network", "nlp", "natural language", "computer vision", "data",
    "data science", "statistics", "probability", "math", "algebra", "calculus",
    "python", "pandas", "numpy", "sklearn", "scikit", "pytorch", "tensorflow",
    "model", "training", "evaluation", "metrics", "overfitting", "underfitting",
    "gradient", "optimization", "regression", "classification", "clustering",
    "prompt", "llm", "gemini", "rnn", "cnn", "transformer","rag","reinforcement learning"
]

def is_study_related(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in STUDY_KEYWORDS)

def get_chat_response(prompt: str, role: str, temperature: float) -> str:
    """Sends a prompt to the persistent Gemini Chat session."""
    try:
        # Get fresh client for this request
        fresh_client = get_gemini_client()
        
        # Create content structure with system instruction
        system_instruction = st.session_state.get("system_instruction", "")
        full_prompt = f"System: {system_instruction}\n\nUser: {prompt}"
        
        # Use simple generate_content approach (like test.py)
        response = fresh_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt,
        )
        
        if response.text:
            return response.text
        return f" {role} returned an empty response."
    except APIError as e:
        return f" Gemini API Error: {str(e)}"
    except Exception as e:
        return f" An unexpected error occurred: {str(e)}"

def reset_and_start_new_chat(system_instruction: str):
    """Resets session state and starts a new Gemini Chat session with a new instruction."""
    st.session_state["messages"] = []
    st.session_state["system_instruction"] = system_instruction
    st.toast(" Chat session reset. New StudyMate role active!")

def get_chat_history_for_download():
    """Formats the session messages into a simple text for download."""
    history = []
    for msg in st.session_state["messages"]:
        history.append(f"--- {msg['role'].upper()} ---\n{msg['content']}\n")
    return "\n\n".join(history).encode("utf-8")

# --- RECOMMENDATION ENGINE (Lightweight, topic-based) ---

# Map topics to suggested follow-up questions
TOPIC_SUGGESTIONS = {
    "neural network": [
        "How do activation functions like ReLU and GELU impact training?",
        "What's the difference between batch norm and layer norm?",
        "Can you explain vanishing/exploding gradients and fixes?",
    ],
    "nlp": [
        "How does tokenization (BPE/WordPiece) work in practice?",
        "What are attention masks and why are they needed?",
        "Compare RNNs, LSTMs, and Transformers for sequence tasks.",
    ],
    "transformer": [
        "Explain self-attention with shapes and complexity analysis.",
        "What is positional encoding and rotary embeddings?",
        "How do encoder-only vs. decoder-only models differ?",
    ],
    "pytorch": [
        "Show a minimal training loop with Dataset/DataLoader.",
        "How to use autograd and avoid common pitfalls?",
        "What are best practices for saving/loading checkpoints?",
    ],
    "tensorflow": [
        "Build a simple Keras model and compile/train it.",
        "How to use tf.data for performant input pipelines?",
        "Tips for debugging NaNs during training.",
    ],
    "statistics": [
        "When to use t-test vs. ANOVA?",
        "Explain bias-variance tradeoff with examples.",
        "What assumptions underlie linear regression?",
    ],
    "optimization": [
        "Compare SGD, Momentum, RMSProp, and Adam.",
        "How to tune learning rate and use schedulers?",
        "When does weight decay help and how to set it?",
    ],
    "evaluation": [
        "Precision/Recall vs. ROC-AUC—when to use which?",
        "Explain cross-validation strategies for imbalanced data.",
        "How to design a robust offline evaluation?",
    ],
}

GENERIC_SUGGESTIONS = [
    "Can you give me a step-by-step study plan on this topic?",
    "What are common pitfalls and best practices here?",
    "Provide a small dataset/code example I can try now.",
]

def pick_topic(text: str) -> str | None:
    if not text:
        return None
    t = text.lower()
    # Prefer more specific keywords first
    ordered = [
        "transformer", "neural network", "nlp", "pytorch", "tensorflow",
        "optimization", "evaluation", "statistics",
    ]
    for k in ordered:
        if k in t:
            return k
    return None

def get_recommendations(context_text: str, limit: int = 3) -> list[str]:
    topic = pick_topic(context_text)
    if topic and topic in TOPIC_SUGGESTIONS:
        return TOPIC_SUGGESTIONS[topic][:limit]
    return GENERIC_SUGGESTIONS[:limit]

# --- STREAMLIT UI ---

st.set_page_config(page_title="AI StudyMate Pro", page_icon="💡", layout="wide")

# Subtle custom styling for chat bubbles and sidebar
st.markdown(
    """
    <style>
    /* Page */
    .block-container {padding-top: 1.5rem;}
    /* Chat message bubbles */
    .stChatMessage.user {background: #f0f7ff; border-radius: 12px; padding: 0.6rem 0.8rem;}
    .stChatMessage.assistant {background: #fafafa; border-radius: 12px; padding: 0.6rem 0.8rem;}
    /* Sidebar headers */
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {margin-top: 0.5rem;}
    /* Recommendation chips */
    .chip {display:inline-block; margin: 0 0.35rem 0.35rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([0.75, 0.25])
with col1:
    st.title("🤖 AI StudyMate Pro")
    st.caption("Your personalized tutor with conversation memory, personas, and smart recommendations.")
with col2:
    st.metric(label="Status", value="Online", delta="Gemini")
st.divider()

# --- SIDEBAR: NEW FEATURE CONTROLS ---

with st.sidebar:
    st.header("⚙️ StudyMate Controls")

    # 1. Role-Play Mode / Persona Switching
    persona = st.selectbox(
        "Select Tutor Persona",
        options=["Academic Tutor", "ELI5 Mode (Simple)", "Technical Interviewer"],
        index=0,
        key="persona_select"
    )

    # Define system instruction based on selection
    if persona == "Academic Tutor":
        instruction = ("You are an engaging and helpful AI StudyMate specializing in AI, ML, and Data Science. "
                       "Your goal is to provide accurate, easy-to-understand, and structured explanations. Use Markdown, lists, and code blocks. "
                       "Only answer questions related to studying AI/ML/Data Science topics. For anything else, ask the user to rephrase.")
    elif persona == "ELI5 Mode (Simple)":
        instruction = ("You are a friendly, concise AI tutor. Explain every technical concept as simply as possible, as if explaining it to a 5-year-old. "
                       "Only answer study-related questions about AI/ML/Data Science; otherwise, ask the user to rephrase.")
    elif persona == "Technical Interviewer":
        instruction = ("You are a strict technical interviewer. Ask challenging follow-up questions to test the user's deep knowledge of AI/ML concepts. Be concise and evaluative. "
                       "Only engage on AI/ML/Data Science topics; otherwise, ask the user to rephrase to a study topic.")

    # 2. Model Parameter Toggles
    # Add a slider to control randomness/creativity
    temperature = st.slider(
        "Creativity (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.2, # Low value is better for factual study content
        step=0.05,
        help="Higher values make the answers more creative/random. Keep low for factual study."
    )
    
    # Study-only restriction toggle
    study_only = st.checkbox(
        "Restrict answers to study-related topics (AI/ML/Data Science)",
        value=True,
        help="When enabled, off-topic prompts will be declined with a helpful message."
    )
    # Show recommendations toggle (persisted)
    st.session_state["show_recs"] = st.checkbox(
        "Show recommended follow-up questions",
        value=st.session_state.get("show_recs", True),
        help="Suggests helpful next questions based on the current topic.",
    )
    
    # Reset Button for the New Chat/Persona
    if st.button("🔄 Start New Chat", type="primary"):
        reset_and_start_new_chat(instruction)

    st.markdown("---")
    st.subheader("📚 Utilities")
    
    # 3. Conversation Export
    if st.session_state["messages"]:
        download_data = get_chat_history_for_download()
        st.download_button(
            label="⬇️ Download Conversation (TXT)",
            data=download_data,
            file_name="StudyMate_Chat_History.txt",
            mime="text/plain",
            key="download_button"
        )
    else:
        st.info("Ask a question to enable download.")

# --- MAIN CHAT INTERFACE ---

# 4. Display chat messages with avatars
for msg in st.session_state["messages"]:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# 5. Input box and Response Generation
# Allow suggestion buttons to set a pending input
user_input = None
if st.session_state.get("pending_user_input"):
    user_input = st.session_state["pending_user_input"]
    st.session_state["pending_user_input"] = None
else:
    user_input = st.chat_input(f"Ask your {persona} anything about AI/ML...")

if user_input:
    # Display user message immediately
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_input)

    # Save user message to history
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Enforce study-only mode before calling the model
    with st.chat_message("assistant", avatar="🧠"):
        if 'study_only' in locals() and study_only and not is_study_related(user_input):
            ai_reply = (
                "I’m focused on study-related topics. Please ask about AI, Machine Learning, Data Science, "
                "math for ML (linear algebra, calculus, probability, statistics), or practical tooling like "
                "Python, NumPy, pandas, scikit-learn, PyTorch, or TensorFlow."
            )
            st.info(ai_reply)
        else:
            # Get response from the persistent chat session
            with st.spinner(f"🧠 {persona} is preparing an answer..."):
                ai_reply = get_chat_response(user_input, persona, temperature)
                st.markdown(ai_reply)

        # Save last assistant reply for recommendations
        st.session_state["last_assistant_reply"] = ai_reply

        # Show recommendations as clickable chips
        if st.session_state.get("show_recs", True):
            st.markdown("---")
            st.subheader("🔎 Recommended next questions")
            recs = get_recommendations(f"{user_input}\n{ai_reply}")
            cols = st.columns(min(3, len(recs))) if recs else []
            for i, rec in enumerate(recs):
                # Arrange buttons in columns
                c = cols[i % max(1, len(cols))] if cols else st
                with c:
                    if st.button(rec, key=f"rec_{i}"):
                        st.session_state["pending_user_input"] = rec
                        st.rerun()

    # Save AI message to history
    st.session_state["messages"].append({"role": "assistant", "content": ai_reply})