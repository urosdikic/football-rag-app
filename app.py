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

    """The most admitted story tells that the game was developed in England in the 12th century. In this century, games that resembled football were played on meadows and roads in England. Besides from kicks, the game involved also punches of the ball with the fist. This early form of football was also much more rough and violent than the modern way of playing. 
    
    Another important event in the history of football came about in 1863 in London when the first Football association was formed in England. It was decided that carrying the ball with the hands wasn't allowed.  The game would, however, continue to develop for a long time and therewas still much flexibility concerning the rules. For one thing, the number of players on the pitch could vary.

    Football clubs have existed since the 15th century, before early clubs were often formed by former school students and the first of this kind was formed in Sheffield in 1855. The oldest among professional football clubs is the English club Notts County that was formed in 1862 and still exists today. 
    
    In the 1880s the interest in the game has moved ahead to a level that tickets were sold to the matches. And finally, in 1885 professional football was legalized and three years later the Football League was established. During the first season, 12 clubs joined the league, but soon more clubs became interested and the competition would consequently expand into more divisions.

    Other milestones were now to follow. Football Association Challenge Cup (FA Cup) became the first important competition when it was run in 1871. The following year, a match between two national teams was played for the first time. The match that involved England and Scotland ended 0-0 and was followed by 4,000 people at Hamilton Crescent. 
    
    The Fédération Internationale de Football Association (FIFA) was founded in 1904 and a foundation act was signed by representatives from France, Belgium, Denmark, the Netherlands, Spain, Sweden and Switzerland. In 1908 would football for the first time be included as an official sport in the Olympic Games. Until the first FIFA World Cup was played in 1930, the Olympic Games football tournament would rank as the most prestigious on a national level. Women's football was not added until 1996.""",

    """The aim of football is to score more goals then your opponent in a 90 minute playing time frame. The match is split up into two halves of 45 minutes. After the first 45 minutes players will take a 15 minute rest period called half time. The second 45 minutes will resume and any time deemed fit to be added on by the referee (injury time) will be accordingly.
    
    To win you have to score more goals than that of your opponents. If the scores are level after 90 minutes then the game will end as a draw apart from in cup games where the game can go to extra time and even a penalty shootout to decide the winner. Players must use their feet to kick the ball and are prohibited to use their hands apart from goalkeepers who can use any part of their body within the 18 yard box.""",

    """•	A match consists of two 45 minutes halves with a 15 minute rest period in between.
    •	Each team can have a minimum off 11 players (including 1 goalkeeper who is the only player allowed to handle the ball within the 18 yard box) and a minimum of 7 players are needed to constitute a match.
    •	Each team can name up to 7 substitute players. Substitutions can be made at any time of the match with each team being able to make a maximum of 3 substitutions per side. In the event of all three substitutes being made and a player having to leave the field for injury the team will be forced to play without a replacement for that player.
    •	Each game must include one referee and two assistant referee’s (linesmen). It’s the job of the referee to act as time keeper and make any decisions which may need to be made such as fouls, free kicks, throw ins, penalties and added on time at the end of each half. The referee may consult the assistant referees at any time in the match regarding a decision.
    •	For fouls committed a player could receive either a yellow or red card depending on the severity of the foul. The yellow is a warning and a red card is a dismissal of that player. Two yellow cards will equal one red. Once a player is sent off then they cannot be replaced.
    •	If a ball goes out of play off an opponent in either of the side lines then it is given as a throw in. If it goes out of play off an attacking player on the base line then it is a goal kick. If it comes off a defending player it is a corner kick.""",

    """The Major International Competitions are:
- FIFA World Cup: Unquestionably the biggest sporting event in the world, held every four years. The next edition will be in 2026, hosted by the USA, Canada, and Mexico.
- UEFA European Championship (Euro): Considered the second strongest international tournament due to the high density of elite national teams.
- Copa América: The oldest continental competition in the world, where South American powerhouses like Argentina and Brazil battle for dominance.

The elite club competitions are:
- UEFA Champions League: The pinnacle of club football, featuring the top teams from Europe's strongest leagues. 
- FIFA Club World Cup: In 2025, this tournament expanded to a new 32-team format, putting it on par with major international tournaments.
- Copa Libertadores: The South American equivalent of the Champions League, known for its incredible passion and gritty style of play.

According to UEFA coefficients and market value, the current top leagues are:
1.	Premier League	England	The most financially powerful and most-watched league globally.
2.	Serie A	Italy	A tactically advanced league that has seen a major resurgence recently.
3.	La Liga	Spain	Home to technical football and some of the world's biggest stars.
4.	Bundesliga	Germany	Famous for the highest stadium attendance and attacking play.
5.	Ligue 1	France	A massive source of talent for all other elite leagues.""",

    """This document lists the results of the FIFA World Cup finals from 1930 to 2010, including the winners, runners-up, final scores, host venues.

- 1930: Uruguay defeated Argentina 4–2 at Estadio Centenario, Uruguay.
- 1934: Italy defeated Czechoslovakia 2–1 (after extra time) at Stadio Nazionale PNF, Italy.
- 1938: Italy defeated Hungary 4–2 at Stade Olympique de Colombes, France.
- 1950: Uruguay defeated Brazil 2–1 at the Maracanã Stadium, Brazil.
- 1954: West Germany defeated Hungary 3–2 at Wankdorf Stadium, Switzerland.
- 1958: Brazil defeated Sweden 5–2 at Råsunda Stadium, Sweden.
- 1962: Brazil defeated Czechoslovakia 3–1 at Estadio Nacional, Chile.
- 1966: England defeated West Germany 4–2 (after extra time) at Wembley Stadium, England.
- 1970: Brazil defeated Italy 4–1 at Estadio Azteca, Mexico.
- 1974: West Germany defeated Netherlands 2–1 at Olympiastadion, West Germany.
- 1978: Argentina defeated Netherlands 3–1 (after extra time) at Estadio Monumental, Argentina.
- 1982: Italy defeated West Germany 3–1 at Santiago Bernabéu, Spain.
- 1986: Argentina defeated West Germany 3–2 at Estadio Azteca, Mexico.
- 1990: West Germany defeated Argentina 1–0 at Stadio Olimpico, Italy.
- 1994: Brazil defeated Italy 0–0 (3–2 on penalties) at the Rose Bowl, USA.
- 1998: France defeated Brazil 3–0 at Stade de France, France.
- 2002: Brazil defeated Germany 2–0 at International Stadium, Japan.
- 2006: Italy defeated France 1–1 (5–3 on penalties) at Olympiastadion, Germany.
- 2010: Spain defeated Netherlands 1–0 (after extra time) at Soccer City, South Africa.""",

    """Offside can be called when an attacking player is in front of the last defender when the pass is played through to them. The offside area is designed to discourage players from simply hanging around the opponent’s goal waiting for a pass. To be onside they must be placed behind the last defender when the ball is played to them. If the player is in front of that last defender then he is deemed to be offside and free kick to the defending team will be called. A player cannot be caught offside in their own half. The goalkeeper does not count as a defender. If the ball is played backwards and the player is in front of the last defender then he is deemed to be not offside.""",

    """Formation, as most fans know, is the arrangement of players on the field. This is typically expressed as a series of numbers (e.g., 4-4-2, 3-5-2) that indicate the number of players in each horizontal line of the tactical setup. It’s the starting point from which coaches build their game plan.
    
    On the other hand, shape refers to how the team looks during play, which is not always consistent with their formation. It’s about the fluidity of players’ positioning, and is greatly influenced by whether the team has possession, where the ball is located, and their intended strategy at any given moment.""",

    """Often, discussions about formation revolve around how many defenders, midfielders, and forwards are deployed. However, understanding the offensive shape can reveal a side’s true intentions and capability. For instance, a 4-3-3 formation can manifest as a staggered diamond in the midfield when the team is in possession. This allows for a strong pressing game and quick, dynamic wing play. Similarly, a 4-4-2 might shift to a flat four in midfield, offering a more balanced approach between attacking and defending.
    
    A team’s defensive shape is equally illuminating; it’s their first line of resistance against the opponent’s advances. Traditionally, a team might drop into a 4-4-2 or a 5-4-1 when under pressure. However, it’s the intricacies of player positioning within these formations that truly dictate the team’s ability to close down spaces and mark opponents. This involves a mix of zonal marking and man-to-man coverage, all while maintaining the integrity of the defensive shape to avoid becoming disjointed.""",

    """WORLD CUP ALL-TIME TOP SCORERS LIST:
1. Miroslav Klose from Germany is the all-time leading goal scorer in World Cup history with 16 goals across 24 appearances.
2. Ronaldo from Brazil is the second highest scorer with 15 goals in 19 appearances.
3. Gerd Müller from West Germany scored 14 goals in only 13 appearances.
4. Lionel Messi from Argentina has scored 13 goals in 26 World Cup appearances.
5. Just Fontaine from France holds the record for most goals in a single tournament, with 13 goals in 6 appearances.""",

    """SLOVENIA NATIONAL TEAM RECORDS: Slovenia has qualified for the FIFA World Cup twice and the UEFA European Championship twice.
SLOVENIA WORLD CUP HISTORY: At the 2010 FIFA World Cup, Slovenia achieved its first and only victory by defeating Algeria 1–0.
SLOVENIA EURO 2024: Slovenia reached the knockout stages of UEFA Euro 2024 for the first time after drawing all three group matches.
SLOVENIA VS ITALY 2004: Slovenia famously defeated Italy 1–0 in 2004, which was Italy's only loss in their entire 2006 World Cup campaign.""",

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
        chunk_size=100,
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
    page = st.radio("Go to:", ["Home", "Search", "Statistics", "Trivia Quiz"])
    
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


# ──────────────────────────────────────────────────────────────────────
# STATISTICS PAGE
# ──────────────────────────────────────────────────────────────────────
elif page == "Statistics":
    st.title("📊 Knowledge Base Analytics")
    st.markdown("Detailed breakdown of the football data currently loaded into the RAG engine.")
    
    # We need the chunks to calculate stats
    vector_store, chunks = build_vector_store(tuple(DOCUMENTS))
    
    # --- ROW 1: Key Metrics in Styled Cards ---
    st.markdown("### Database Overview")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        
        # Total source documents
        col1.metric("Source Docs", len(DOCUMENTS))
        
        # Total chunks created
        col2.metric("Vector Chunks", len(chunks))
        
        # Average characters per chunk
        avg_len = sum(len(c) for c in chunks) / len(chunks)
        col3.metric("Avg Chunk Size", f"{avg_len:.0f} chars")

    st.divider()

    # --- ROW 2: The Chart ---
    st.subheader("Chunk Size Distribution")
    chunk_lengths = [len(c) for c in chunks]
    st.bar_chart(chunk_lengths, color="#2e7d32") # Matching the green football theme

    # --- ROW 3: Keywords ---
    st.subheader("Top Keywords in Database")
    all_text = " ".join(DOCUMENTS).lower()
    # Filter out common stop words and punctuation
    words = [w.strip(".,!-") for w in all_text.split() if len(w) > 4]
    
    import collections
    word_counts = collections.Counter(words).most_common(10)
    
    import pandas as pd
    df_words = pd.DataFrame(word_counts, columns=['Keyword', 'Frequency'])
    st.table(df_words)

# ──────────────────────────────────────────────────────────────────────
# TRIVIA QUIZ PAGE
# ──────────────────────────────────────────────────────────────────────

elif page == "Trivia Quiz":
    st.title("🏆 Football History Quiz")
    st.markdown("Test your knowledge! All these facts are hidden in our database.")

    # 1. Initialize a quiz version ID to force-reset widgets
    if 'quiz_id' not in st.session_state:
        st.session_state.quiz_id = 0

    current_score = 0

    # Question 1 - Notice the unique key using quiz_id
    st.subheader("Question 1")
    q1 = st.radio(
        "Who won the very first FIFA World Cup in 1930?",
        ["Brazil", "Argentina", "Uruguay", "Germany"],
        index=None,
        key=f"q1_{st.session_state.quiz_id}"
    )
    
    if q1 == "Uruguay":
        st.success("Correct! Uruguay won the inaugural FIFA World Cup in 1930, defeating Argentina 4-2 in the final.")
        current_score += 1
    elif q1 is not None:
        st.error("Not quite! Check the 'Search' page to find the answer.")

    st.divider()

    # Question 2
    st.subheader("Question 2")
    q2 = st.radio(
        "In which year did Slovenia achieve its first World Cup victory?",
        ["2002", "2010", "2014", "Never"],
        index=None,
        key=f"q2_{st.session_state.quiz_id}"
    )

    if q2 == "2010":
        st.success("Correct! Matjaž Kek's team beat Algeria 1-0.")
        current_score += 1
    elif q2 is not None:
        st.error("Incorrect. (Hint: It was in South Africa!)")

    st.divider()

    # Question 3
    st.subheader("Question 3")
    q3 = st.radio(
        "Which country hosted the 1990 FIFA World Cup?",
        ["Italy", "Germany", "Brazil", "Argentina"],
        index=None,
        key=f"q3_{st.session_state.quiz_id}"
    )

    if q3 == "Italy":
        st.success("Correct! The 1990 FIFA World Cup was hosted by Italy.")
        current_score += 1
    elif q3 is not None:
        st.error("Wrong answer! Check the 'Search' page to find the answer.")

    st.divider()

    # Question 4
    st.subheader("Question 4")
    q4 = st.radio(
        "Who is the best goal scorer in World Cup history?",
        ["Pelé", "Lionel Messi", "Miroslav Klose", "Cristiano Ronaldo"],
        index=None,
        key=f"q4_{st.session_state.quiz_id}"
    )

    if q4 == "Miroslav Klose":
        st.success("Correct! Miroslav Klose is the all-time leading goal scorer in World Cup history.")
        current_score += 1
    elif q4 is not None:
        st.error("Wrong answer! Check the 'Search' page to find the answer.")

    # Score Display in Sidebar
    st.sidebar.metric("Your Quiz Score", f"{current_score} / 4")
    
    # Final Score and Reset
    st.divider()
    
    if st.button("Reset Quiz", key="final_reset"):
        # Change the ID to force all radio buttons to reset
        st.session_state.quiz_id += 1
        st.rerun()


st.markdown("---")
footer_col1, footer_col2 = st.columns(2)
with footer_col1:
    st.caption("© 2026 Football RAG System | AI Course")
with footer_col2:
    st.markdown("""
        <div style="text-align: right;">
            <a href="https://github.com/urosdikic/football-rag-app" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-View_Source-181717?logo=github" />
            </a>
        </div>
    """, unsafe_allow_html=True)

#

