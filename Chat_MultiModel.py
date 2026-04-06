import streamlit as st
import io
from bytez import Bytez
from google import genai
from google.genai.errors import APIError
import os
from dotenv import load_dotenv

# --- CONFIGURATION ---

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BYTEZ_API_KEY = os.getenv("BYTEZ_API_KEY")

if not GEMINI_API_KEY:
    st.error(" GEMINI_API_KEY not found!")
if not BYTEZ_API_KEY:
    st.error(" BYTEZ_API_KEY not found!")

# Cache API clients
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

@st.cache_resource
def get_bytez_client():
    return Bytez(BYTEZ_API_KEY)

# --- QUOTA STATUS CHECKER ---

def check_quota_status() -> dict:
    """Check quota status for both models"""
    status = {
        "gemma": "unknown",
        "gemini": "unknown"
    }
    
    # Quick test for Gemma
    try:
        bytez_client = get_bytez_client()
        model = bytez_client.model("google/gemma-3-4b-it")
        result = model.run([{"role": "user", "content": "Hi"}])
        if result.error and ("429" in str(result.error) or "quota" in str(result.error).lower()):
            status["gemma"] = "exceeded"
        else:
            status["gemma"] = "available"
    except:
        status["gemma"] = "error"
    
    # Quick test for Gemini
    try:
        gemini_client = get_gemini_client()
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hi"
        )
        status["gemini"] = "available"
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
            status["gemini"] = "exceeded"
        else:
            status["gemini"] = "error"
    
    return status

# --- SESSION STATE ---

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "system_instruction" not in st.session_state:
    st.session_state["system_instruction"] = (
        "You are an expert AI tutor and educator specializing in AI, Machine Learning, and Data Science. "
        "Your responses must be natural, conversational, and engaging, just like ChatGPT or Claude. "
        "Guidelines: "
        "1. Be conversational and friendly, not robotic or overly formal "
        "2. Provide comprehensive yet accessible explanations "
        "3. Use natural language patterns and flow "
        "4. Include relevant examples and analogies naturally "
        "5. Structure responses logically with clear headings when needed "
        "6. Use markdown formatting effectively but keep it clean and readable "
        "7. Ask follow-up questions to encourage continued learning "
        "8. Match the quality and style of top-tier LLMs like ChatGPT "
        "9. Be helpful and encouraging in your tone "
        "10. Focus exclusively on AI/ML/Data Science topics "
        "Your responses should feel like a knowledgeable expert having a natural conversation, not a formatted template."
    )

if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = "gemma-3-4b-it"

if "quota_status" not in st.session_state:
    st.session_state["quota_status"] = check_quota_status()

# --- CORE FUNCTIONALITY ---

# Study keywords for relevance checking
STUDY_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "ml", "deep learning",
    "neural network", "nlp", "natural language", "computer vision", "data",
    "data science", "statistics", "probability", "math", "algebra", "calculus",
    "python", "pandas", "numpy", "sklearn", "scikit", "pytorch", "tensorflow",
    "model", "training", "evaluation", "metrics", "overfitting", "underfitting",
    "gradient", "optimization", "regression", "classification", "clustering",
    "prompt", "llm", "gemini", "gpt", "rnn", "cnn", "transformer","rag","reinforcement learning"
]

def is_study_related(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in STUDY_KEYWORDS)

def format_llm_response(raw_response: str, question: str) -> str:
    """Formats raw LLM response to match the desired structure while preserving the actual answer."""
    formatted_response = []
    
    # Extract the main answer from raw response (simplified parsing)
    # Look for key content that answers the question
    lines = raw_response.split('\n')
    main_answer = ""
    
    # Try to find a good definition/explanation in the response
    for line in lines:
        if line.strip() and not line.startswith('#') and not line.startswith('*') and not line.startswith('-'):
            if len(line.strip()) > 20:  # Substantial content
                main_answer = line.strip()
                break
    
    # If no good answer found, use first substantial paragraph
    if not main_answer:
        for line in lines:
            if line.strip() and len(line.strip()) > 30:
                main_answer = line.strip()
                break
    
    # If still no answer, create one based on the question
    if not main_answer:
        if "llm" in question.lower():
            main_answer = "An LLM (Large Language Model) is a type of AI model that understands and generates human-like text using deep learning techniques."
        elif "rag" in question.lower():
            main_answer = "RAG (Retrieval-Augmented Generation) is an AI technique that enhances LLM responses by retrieving relevant information from a knowledge base before generating answers."
        elif "nlp" in question.lower():
            main_answer = "NLP (Natural Language Processing) is a field of AI that enables computers to understand, interpret, and generate human language."
        else:
            main_answer = raw_response[:200] + "..." if len(raw_response) > 200 else raw_response
    
    # Create the formatted response
    question_lower = question.lower()
    
    if "llm" in question_lower:
        formatted_response.append("## 🧠 What is an LLM?")
        formatted_response.append(main_answer)
        formatted_response.append("")
        formatted_response.append("👍 In simple words:")
        formatted_response.append("LLM = AI that can read, write, and understand language like humans")
        formatted_response.append("")
        formatted_response.append("---")
        formatted_response.append("## 🔍 Real Examples of LLMs")
        formatted_response.append("* ChatGPT")
        formatted_response.append("* Gemini")
        formatted_response.append("* LLaMA")
        formatted_response.append("")
        formatted_response.append("These models can:")
        formatted_response.append("* Answer questions")
        formatted_response.append("* Write essays/code")
        formatted_response.append("* Translate languages")
        formatted_response.append("* Chat like humans")
    
    elif "rag" in question_lower:
        formatted_response.append("## 🔍 What is RAG?")
        formatted_response.append(main_answer)
        formatted_response.append("")
        formatted_response.append("👍 In simple words:")
        formatted_response.append("RAG = AI that finds relevant info first, then answers using that info")
        formatted_response.append("")
        formatted_response.append("---")
        formatted_response.append("## 🎯 RAG Components")
        formatted_response.append("* Knowledge Base (documents)")
        formatted_response.append("* Retrieval System (search)")
        formatted_response.append("* LLM (answer generation)")
        formatted_response.append("")
        formatted_response.append("RAG helps AI:")
        formatted_response.append("* Give accurate, up-to-date answers")
        formatted_response.append("* Avoid making things up")
        formatted_response.append("* Use specific sources")
    
    elif "nlp" in question_lower:
        formatted_response.append("## 💬 What is NLP?")
        formatted_response.append(main_answer)
        formatted_response.append("")
        formatted_response.append("👍 In simple words:")
        formatted_response.append("NLP = AI that understands and processes human language")
        formatted_response.append("")
        formatted_response.append("---")
        formatted_response.append("## 🔧 NLP Tasks")
        formatted_response.append("* Text classification")
        formatted_response.append("* Sentiment analysis")
        formatted_response.append("* Language translation")
        formatted_response.append("* Named entity recognition")
        formatted_response.append("")
        formatted_response.append("NLP is used in:")
        formatted_response.append("* Chatbots")
        formatted_response.append("* Email filters")
        formatted_response.append("* Voice assistants")
    
    else:
        # Generic format for other questions
        formatted_response.append(f"## 🤖 Answer to: {question}")
        formatted_response.append(main_answer)
        formatted_response.append("")
        formatted_response.append("---")
        formatted_response.append("## 💡 Key Points")
        # Extract some key points from the response
        for line in lines[:5]:
            if line.strip() and (line.startswith('*') or line.startswith('-') or line.startswith('•')):
                formatted_response.append(line)
    
    return "\n".join(formatted_response)

def get_gemma_response(prompt: str) -> str:
    """Get response from Gemma-3-4b-it via Bytez"""
    try:
        bytez_client = get_bytez_client()
        model = bytez_client.model("google/gemma-3-4b-it")
        
        # Simple approach: Only use current prompt with system instruction
        system_instruction = st.session_state.get("system_instruction", "")
        
        # Create clean message structure without conversation history for now
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        
        # Run model
        result = model.run(messages)
        
        if result.error:
            if "429" in str(result.error) or "quota" in str(result.error).lower():
                return "⚠️ Gemma quota exceeded. The free tier limit has been reached. Please:\n• Wait for quota reset (usually daily)\n• Try Gemini model instead\n• Upgrade your plan for higher limits"
            return f" Gemma Error: {result.error}"
        
        # Return natural response without strict formatting
        raw_response = result.output or " No response received from Gemma."
        
        # Handle different response types
        if isinstance(raw_response, dict):
            # If it's a dict, try to get the content
            raw_response = raw_response.get('content', str(raw_response))
        elif isinstance(raw_response, list):
            # If it's a list, join it
            raw_response = ' '.join(str(item) for item in raw_response)
        elif not isinstance(raw_response, str):
            # Convert to string if it's not already
            raw_response = str(raw_response)
            
        return raw_response
        
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            return "⚠️ Gemma quota exceeded. The free tier limit has been reached. Please:\n• Wait for quota reset (usually daily)\n• Try Gemini model instead\n• Upgrade your plan for higher limits"
        return f" Gemma Error: {error_str}"

def get_gemini_response(prompt: str) -> str:
    """Get response from Gemini"""
    try:
        gemini_client = get_gemini_client()
        
        # Create content structure with system instruction
        system_instruction = st.session_state.get("system_instruction", "")
        full_prompt = f"System: {system_instruction}\n\nUser: {prompt}"
        
        # Use simple generate_content approach
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt,
        )
        
        if response.text:
            return response.text
        return " Gemini returned an empty response."
        
    except APIError as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
            return "⚠️ Gemini quota exceeded. The free tier limit has been reached. Please:\n• Wait for quota reset (usually daily)\n• Try GPT-4o model instead\n• Upgrade your plan for higher limits"
        return f" Gemini API Error: {error_str}"
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
            return "⚠️ Gemini quota exceeded. The free tier limit has been reached. Please:\n• Wait for quota reset (usually daily)\n• Try GPT-4o model instead\n• Upgrade your plan for higher limits"
        return f" Gemini Error: {error_str}"

def get_chat_response(prompt: str, model: str) -> str:
    """Get response from selected model"""
    if model == "gemma-3-4b-it":
        return get_gemma_response(prompt)
    else:
        return get_gemini_response(prompt)

def reset_and_start_new_chat(system_instruction: str):
    """Reset session state with new system instruction"""
    st.session_state["messages"] = []
    st.session_state["system_instruction"] = system_instruction
    st.toast("✅ Chat session reset. New StudyMate role active!")

def get_chat_history_for_download():
    """Format session messages for download"""
    history = []
    for msg in st.session_state["messages"]:
        history.append(f"--- {msg['role'].upper()} ---\n{msg['content']}\n")
    return "\n\n".join(history).encode("utf-8")

# --- RECOMMENDATION ENGINE ---

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
    "python": [
        "Show a minimal training loop with Dataset/DataLoader.",
        "How to use autograd and avoid common pitfalls?",
        "What are best practices for saving/loading checkpoints?",
    ],
    "statistics": [
        "When to use t-test vs. ANOVA?",
        "Explain bias-variance tradeoff with examples.",
        "What assumptions underlie linear regression?",
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
    ordered = [
        "transformer", "neural network", "nlp", "python", "statistics",
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

st.set_page_config(page_title="AI StudyMate Pro - Multi Model", page_icon="🤖", layout="wide")

# Custom styling
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    .stChatMessage.user {background: #f0f7ff; border-radius: 12px; padding: 0.6rem 0.8rem;}
    .stChatMessage.assistant {background: #fafafa; border-radius: 12px; padding: 0.6rem 0.8rem;}
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {margin-top: 0.5rem;}
    .chip {display:inline-block; margin: 0 0.35rem 0.35rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([0.75, 0.25])
with col1:
    st.title("🤖 AI StudyMate Pro - Multi Model")
    st.caption("Your AI tutor with Gemma and Gemini models, conversation memory, and smart recommendations.")
with col2:
    model_name = "Gemma-3-4b-it" if st.session_state["selected_model"] == "gemma-3-4b-it" else "Gemini 2.0 Flash"
    st.metric(label="Active Model", value=model_name)
st.divider()

# --- SIDEBAR CONTROLS ---

with st.sidebar:
    st.header("⚙️ StudyMate Controls")
    
    # Quota Status Display
    st.subheader("📊 API Status")
    quota_status = st.session_state.get("quota_status", {"gemma": "unknown", "gemini": "unknown"})
    
    gemma_status = quota_status.get("gemma", "unknown")
    gemini_status = quota_status.get("gemini", "unknown")
    
    # Gemma Status
    if gemma_status == "available":
        st.success("🔮 Gemma: Available")
    elif gemma_status == "exceeded":
        st.error("🔮 Gemma: Quota Exceeded")
    else:
        st.warning("🔮 Gemma: Error")
    
    # Gemini Status
    if gemini_status == "available":
        st.success("🧠 Gemini: Available")
    elif gemini_status == "exceeded":
        st.error("🧠 Gemini: Quota Exceeded")
    else:
        st.warning("🧠 Gemini: Error")
    
    # Refresh quota status button
    if st.button("🔄 Check Quota Status"):
        st.session_state["quota_status"] = check_quota_status()
        st.rerun()
    
    st.markdown("---")
    
    # Model Selection
    selected_model = st.selectbox(
        "Select AI Model",
        options=["gemma-3-4b-it", "gemini-2.0-flash"],
        format_func=lambda x: "Gemma-3-4b-it (via Bytez)" if x == "gemma-3-4b-it" else "Gemini 2.0 Flash",
        index=0 if st.session_state["selected_model"] == "gemma-3-4b-it" else 1,
        key="model_select"
    )
    st.session_state["selected_model"] = selected_model
    
    # Role-Play Mode
    persona = st.selectbox(
        "Select Tutor Persona",
        options=["Academic Tutor", "ELI5 Mode (Simple)", "Technical Interviewer"],
        index=0,
        key="persona_select"
    )
    
    # Define system instruction based on selection
    if persona == "Academic Tutor":
        instruction = (
            "You are a world-class AI professor and tutor from a top university like MIT or Stanford. "
            "Your teaching style combines rigorous academic depth with exceptional clarity. "
            "Provide comprehensive explanations with: "
            "• Clear conceptual foundations first, then practical applications "
            "• Mathematical intuition behind algorithms and models "
            "• Industry-relevant code examples and best practices "
            "• Connections to current research and real-world implementations "
            "• Structured learning progression from basics to advanced topics "
            "Your responses should match the quality of premier educational platforms while being more personalized and interactive."
        )
    elif persona == "ELI5 Mode (Simple)":
        instruction = (
            "You are an expert at simplifying complex AI/ML concepts with remarkable clarity. "
            "Your approach: "
            "• Start with simple, intuitive analogies anyone can understand "
            "• Gradually build up to technical details without overwhelming "
            "• Use everyday examples and metaphors effectively "
            "• Keep explanations concise but complete "
            "• Focus on core intuition rather than mathematical rigor "
            "• Make learning feel natural and effortless "
            "Your simplified explanations should be as clear as top-tier educational content but more accessible."
        )
    elif persona == "Technical Interviewer":
        instruction = (
            "You are a senior AI engineer from a leading tech company (Google, OpenAI, Meta) conducting technical interviews. "
            "Your style: "
            "• Ask precise, challenging questions that test deep understanding "
            "• Expect thorough, well-structured technical responses "
            "• Probe for practical implementation knowledge and trade-offs "
            "• Demand code-quality awareness and best practices "
            "• Evaluate problem-solving approach and communication clarity "
            "• Provide expert-level feedback on responses "
            "Maintain the rigor of FAANG technical interviews while being educational."
        )
    
    # Study-only restriction toggle
    study_only = st.checkbox(
        "Restrict answers to study-related topics (AI/ML/Data Science)",
        value=True,
        help="When enabled, off-topic prompts will be declined with a helpful message."
    )
    
    # Show recommendations toggle
    st.session_state["show_recs"] = st.checkbox(
        "Show recommended follow-up questions",
        value=st.session_state.get("show_recs", True),
        help="Suggests helpful next questions based on the current topic.",
    )
    
    # Reset Button
    if st.button("🔄 Start New Chat", type="primary"):
        reset_and_start_new_chat(instruction)
    
    st.markdown("---")
    st.subheader("📚 Utilities")
    
    # Conversation Export
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

# Display chat messages
for msg in st.session_state["messages"]:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🔮" if st.session_state["selected_model"] == "gemma-3-4b-it" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Input box and Response Generation
user_input = None
if st.session_state.get("pending_user_input"):
    user_input = st.session_state["pending_user_input"]
    st.session_state["pending_user_input"] = None
else:
    user_input = st.chat_input(f"Ask your {persona} anything about AI/ML...")

if user_input:
    # Display user message
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_input)
    
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Generate response
    with st.chat_message("assistant", avatar="🔮" if st.session_state["selected_model"] == "gemma-3-4b-it" else "🧠"):
        if study_only and not is_study_related(user_input):
            ai_reply = (
                "I'm focused on study-related topics. Please ask about AI, Machine Learning, Data Science, "
                "math for ML (linear algebra, calculus, probability, statistics), or practical tooling like "
                "Python, NumPy, pandas, scikit-learn, PyTorch, or TensorFlow."
            )
            st.info(ai_reply)
        else:
            # Get response from selected model
            model_name = "Gemma-3-4b-it" if st.session_state["selected_model"] == "gemma-3-4b-it" else "Gemini"
            with st.spinner(f"🔮 {model_name} {persona} is preparing an answer..."):
                ai_reply = get_chat_response(user_input, st.session_state["selected_model"])
                st.markdown(ai_reply)
        
        # Save for recommendations
        st.session_state["last_assistant_reply"] = ai_reply
        
        # Show recommendations
        if st.session_state.get("show_recs", True):
            st.markdown("---")
            st.subheader("🔎 Recommended next questions")
            recs = get_recommendations(f"{user_input}\n{ai_reply}")
            cols = st.columns(min(3, len(recs))) if recs else []
            for i, rec in enumerate(recs):
                c = cols[i % max(1, len(cols))] if cols else st
                with c:
                    if st.button(rec, key=f"rec_{i}"):
                        st.session_state["pending_user_input"] = rec
                        st.rerun()
    
    # Save AI message
    st.session_state["messages"].append({"role": "assistant", "content": ai_reply})
