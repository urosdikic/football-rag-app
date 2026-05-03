import streamlit as st
import numpy as np

st.set_page_config(
    page_title="My RAG-Football-Knowledge Base",
    page_icon="⚽",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────
# YOUR DOCUMENTS — Replace these with your own topic!
# Each string is one "document" that will be chunked, embedded, and
# stored in the vector database for semantic search.
# ──────────────────────────────────────────────────────────────────────
DOCUMENTS = [

    """The most admitted story tells that the game was developed in England in the 12th century. In this century, games that resembled football were played on meadows and roads in England. Besides from kicks, the game involved also punches of the ball with the fist. The oldest among professional football clubs is the English club Notts County that was formed in 1862 and still exists today. 
    
    The Fédération Internationale de Football Association (FIFA) was founded in 1904 and a foundation act was signed by representatives from France, Belgium, Denmark, the Netherlands, Spain, Sweden and Switzerland. In 1908 would football for the first time be included as an official sport in the Olympic Games. Until the first FIFA World Cup was played in 1930, the Olympic Games football tournament would rank as the most prestigious on a national level.""",

    """The aim of football is to score more goals then your opponent in a 90 minute playing time frame. The match is split up into two halves of 45 minutes. After the first 45 minutes players will take a 15 minute rest period called half time. The second 45 minutes will resume and any time deemed fit to be added on by the referee (injury time) will be accordingly.
    
    To win you have to score more goals than that of your opponents. If the scores are level after 90 minutes then the game will end as a draw. Players must use their feet to kick the ball and are prohibited to use their hands apart from goalkeepers who can use any part of their body within the 18 yard box.""",

    """•	Each team can have a minimum off 11 players (including 1 goalkeeper who is the only player allowed to handle the ball within the 18 yard box) and a minimum of 7 players are needed to constitute a match.
    •	Each team can name up to 7 substitute players. Substitutions can be made at any time of the match with each team being able to make a maximum of 3 substitutions per side. In the event of all three substitutes being made and a player having to leave the field for injury the team will be forced to play without a replacement for that player.
    •	For fouls committed a player could receive either a yellow or red card depending on the severity of the foul. The yellow is a warning and a red card is a dismissal of that player. Two yellow cards will equal one red. Once a player is sent off then they cannot be replaced."""

]

# ──────────────────────────────────────────────────────────────────────
# Cached heavy resources (loaded once, reused across reruns)
# ──────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Building vector database...")
def build_vector_store(_documents: list    ):
    """Chunk documents, embed them, and store in ChromaDB."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    # --- Chunking ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=10,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in _documents:
        chunks.extend(splitter.split_text(doc))

    embeddings = load_embedding_model()

    # --- Store in ChromaDB ---
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="knowledge_base",
    )
    return vector_store, chunks


# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────
# 1. Add the Branding Box at the very top
with st.sidebar:
    st.markdown("""
        <div style="background-color: #2e7d32; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 25px;">
            <h1 style="color: white; margin: 0; font-size: 1.5rem;">⚽ Football RAG</h1>
        </div>
    """, unsafe_allow_html=True)

    # 2. Your existing navigation
    st.title("Navigation")
    page = st.radio("Go to:", ["Home", "Search"])
    
    # 3. Add a little "Status" indicator at the bottom of the sidebar
    st.divider()
    st.markdown("### 🟢 System Status")
    st.caption("Model: MiniLM-L6-v2")
    st.caption("Database: ChromaDB (Active)")

# ──────────────────────────────────────────────────────────────────────
# HOME PAGE
# ──────────────────────────────────────────────────────────────────────

if page == "Home":
    st.title("⚽ Football Intelligence RAG")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About the Knowledge Base
        This AI-powered tool uses **Semantic Search** to explore the history, rules, and statistics of world football. 
        Unlike traditional search, this app understands the *meaning* behind your questions.
        """)
        st.info("💡 **Pro Tip:** Try asking about the 'Origins of the game' or 'World Cup winners'.")

    with col2:
        st.success(f"📈 **Database Stats**\n\n- {len(DOCUMENTS)} Documents\n- Vector Engine: ChromaDB\n- Model: MiniLM-L6-v2")

# ──────────────────────────────────────────────────────────────────────
# SEARCH PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "Search":
    st.title("🔍 Semantic Search")
    st.markdown("Ask a question and the app will find the most relevant information.")

    # 1. ADD THIS: Example Question Logic
    st.markdown("#### 💡 Quick Search Examples:")
    cols = st.columns(3)
    
    # These buttons update the "search_query" in session state
    if cols[0].button("🏅 All-time Top Scorers"):
        st.session_state.search_query = "Who are the top 10 goal scorers in World Cup history?"
        st.rerun()
    if cols[1].button("🇸🇮 Slovenia National Team"):
        st.session_state.search_query = "Slovenia national team records"
        st.rerun()
    if cols[2].button("📏 Offside Rule"):
        st.session_state.search_query = "Explain the offside rule in football."
        st.rerun()

    # 2. UPDATE THE TEXT INPUT: It now looks at session_state for its value
    query = st.text_input(
        "Your question",
        value=st.session_state.get('search_query', ''), # This links the buttons to the box
        placeholder="e.g. Tell me something about the history of football.",
        key="main_search_input"
    )

    num_results = st.slider("Number of results", 1, 10, 3)

    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))

    if query:
        with st.spinner("Searching..."):
            results = vector_store.similarity_search_with_score(query, k=num_results)

        st.subheader(f"Top {len(results)} results")
        for i, (doc, score) in enumerate(results, 1):
            similarity = max(0, 1 - score)
            with st.expander(f"📍 Match {i} (Relevance: {similarity:.2f})", expanded=True):
                st.write(doc.page_content)
                st.caption(f"Source Document Chunk — Size: {len(doc.page_content)} characters")

    st.markdown("---")
    st.caption("Powered by all-MiniLM-L6-v2 embeddings + ChromaDB")

#

