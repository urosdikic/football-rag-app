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
    """The first known examples of a team game involving a ball, which was made out of a rock, occurred in old Mesoamerican cultures for over 3000 years ago. It was by the Aztecs called Tchatali, although various versions of the game were spread over large regions. In some ritual occasions, the ball would symbolize the sun and the captain of the losing team would be sacrificed to the gods. A unique feature of the Mesoamerican ball game versions was a bouncing ball made of rubber – no other early culture had access to rubber. The first known ball game which also involved kicking took place In China in the 3rd and 2nd century BC under the name cuju. Cuju was played with a round ball (stitched leather with fur or feathers inside) on an area of a square. A modified form of this game later spread to Japan and was by the name of kemari practiced under ceremonial forms. Perhaps even older than cuju was Marn Gook, played by Aboriginal Australians and according to white emigrants in the 1800s, a ball game that primarily involving kicking. The ball was made by encased leaves or roots. The rules are mostly unknown, but as with many other early versions of the game, a chief feature seems to have been to keep the ball in the air. Other variety of ball games had been known from Ancient Greece. The ball was made by shreds of leather filled with hair (the first documents of balls filled with air are from the 7th century). Ball games had, however, a low status and was not included at the Panhellenic Games. In the Ancient Rome, games with balls were not included in the entertainment on the big arenas (amphitheaters), but occurred in exercises in the military by the name of Harpastum. It was the Roman culture that would bring football to the British island (Britannica). It is, however, uncertain in which degree the British people were influenced by this variety and in which degree they had developed their own variants.""",

    """The most admitted story tells that the game was developed in England in the 12th century. In this century, games that resembled football were played on meadows and roads in England. Besides from kicks, the game involved also punches of the ball with the fist. This early form of football was also much more rough and violent than the modern way of playing. An important feature of the forerunners to football was that the games involved plenty of people and took place over large areas in towns (an equivalent was played in Florence from the 16th century where it was called Calcio). The rampage of these games would cause damage on the town and sometimes death to the participants. These would be among the reasons for the proclamations against the game that finally was forbidden for several centuries. But the football-like games would return to the streets of London in the 17th century. It would be forbidden again in 1835, but at this stage the game had been established in the public schools. An attempt to create proper rules for the game was done at a meeting in Cambridge in 1848, but a final solution to all questions of rules was not achieved. Another important event in the history of football came about in 1863 in London when the first Football association was formed in England. It was decided that carrying the ball with the hands wasn't allowed. The meeting also resulted in a standardization of the size and weight of the ball. A consequence of the London meeting was that the game was divided into two codes: association football and rugby. The game would, however, continue to develop for a long time and therewas still much flexibility concerning the rules. For one thing, the number of players on the pitch could vary. Neither were uniforms used to distinguish the appearance of the teams. It was also common with players wearing caps – the header was yet to be a part of the game yet. Another important difference at this stage could be noticed between English and Scottish teams. Whereas the English teams preferred to run forward with the ball in a more rugby fashion, the Scottish chose to pass the ball between their players. It would be the Scottish approach that soon became predominant. The sport was at first an entertainment for the British working class. Unprecedented amounts of spectators, up to 30,000, would see the big matches in the late 19th century. The game would soon expand by British peoples who traveled to other parts of the world and as a result to the British colonization efforts. Especially in South America and India would the interest in football become big.""",

    """Football clubs have existed since the 15th century, but unorganized and without official status. It is therefore hard to decide which the first football club was. Some historians suggest that it was the Foot-Ball Club formed 1824 in Edinburgh. Early clubs were often formed by former school students and the first of this kind was formed in Sheffield in 1855. The oldest among professional football clubs is the English club Notts County that was formed in 1862 and still exists today. An important step for the emergence of teams was the industrialization that led to larger groups of people meeting at places such as factories, pubs and churches. Football teams were established in the larger cities and the new railroads could bring them to other cities. In the beginning, football was dominated by public school teams, but later, teams consisting by workers would make up the majority. Another change was successively taking place when some clubs became willing to pay the best players to join their team. This would be the start of a long period of transition, not without friction, in which the game would develop to a professional level. The motivation behind paying players was not only to win more matches. In the 1880s the interest in the game has moved ahead to a level that tickets were sold to the matches. And finally, in 1885 professional football was legalized and three years later the Football League was established. During the first season, 12 clubs joined the league, but soon more clubs became interested and the competition would consequently expand into more divisions. For a long time, the British teams would be dominant. After some decades, clubs from Prague, Budapest and Sienna would be the primarily contenders to the British dominance. As with many things in history, women were for a long time excluded from participating in games. It was not before the late 19th century that women started to play football. The first official women's game took place in Inverness in 1888.""",

    """Other milestones were now to follow. Football Association Challenge Cup (FA Cup) became the first important competition when it was run in 1871. The following year, a match between two national teams was played for the first time. The match that involved England and Scotland ended 0-0 and was followed by 4,000 people at Hamilton Crescent (the picture shows illustrations from this occasion). Twelve years later, in 1883, the first international tournament took place and included four national teams: England, Ireland, Scotland and Wales. Football was for a long time a British phenomenon, but it gradually spread to other European countries. The first game that took place outside Europe occurred in Argentina in 1867, but it was foreign British workers who were involved and not Argentinean citizens. The Fédération Internationale de Football Association (FIFA) was founded in 1904 and a foundation act was signed by representatives from France, Belgium, Denmark, the Netherlands, Spain, Sweden and Switzerland. England and the other British countries did not join FIFA from the start, they had invented the game and saw no reason to subordinate to an association. Still, they joined in the following year, but would not partake in the World Cup until 1950. Domestic leagues occurred in many countries. The first was, as already mentioned, the English Football League which was established in 1888. The leagues would by time expand by more divisions, which were based on team performance. In 1908 would football for the first time be included as an official sport in the Olympic Games. Until the first FIFA World Cup was played in 1930, the Olympic Games football tournament would rank as the most prestigious on a national level. Women's football was not added until 1996.""",

    """The aim of football is to score more goals then your opponent in a 90 minute playing time frame. The match is split up into two halves of 45 minutes. After the first 45 minutes players will take a 15 minute rest period called half time. The second 45 minutes will resume and any time deemed fit to be added on by the referee (injury time) will be accordingly.

    Each team consists of 11 players. These are made up of one goalkeeper and ten outfield players. The pitch dimensions vary from each ground but are roughly 120 yards long and 75 yards wide. On each pitch you will have a 6 yard box next to the goal mouth, an 18 yard box surrounding the 6 yard box and a centre circle. Each half of the pitch must be a mirror image of the other in terms of dimensions.
    
    Essentially the equipment that is needed for a soccer match is pitch and a football. Additionally players can be found wearing studded football boots, shin pads and matching strips. The goalkeepers will additionally wear padded gloves as they are the only players allowed to handle the ball. Each team will have a designated captain.
    
    To score the ball must go into your opponent’s goal. The whole ball needs to be over the line for it to be a legitimate goal. A goal can be scored with any part of the body apart from the hand or arm up to the shoulder. The goal itself consists of a frame measuring 8 feet high and 8 yards wide.
    
    To win you have to score more goals than that of your opponents. If the scores are level after 90 minutes then the game will end as a draw apart from in cup games where the game can go to extra time and even a penalty shootout to decide the winner. Players must use their feet to kick the ball and are prohibited to use their hands apart from goalkeepers who can use any part of their body within the 18 yard box.""",

    """•	A match consists of two 45 minutes halves with a 15 minute rest period in between.
    •	Each team can have a minimum off 11 players (including 1 goalkeeper who is the only player allowed to handle the ball within the 18 yard box) and a minimum of 7 players are needed to constitute a match.
    •	The field must be made of either artificial or natural grass. The size of pitches is allowed to vary but must be within 100-130 yards long and 50-100 yards wide. The pitch must also be marked with a rectangular shape around the outside showing out of bounds, two six yard boxes, two 18 yard boxes and a centre circle. A spot for a penalty placed 12 yards out of both goals and centre circle must also be visible.
    •	The ball must have a circumference of 58-61cm and be of a circular shape.
    •	Each team can name up to 7 substitute players. Substitutions can be made at any time of the match with each team being able to make a maximum of 3 substitutions per side. In the event of all three substitutes being made and a player having to leave the field for injury the team will be forced to play without a replacement for that player.
    •	Each game must include one referee and two assistant referee’s (linesmen). It’s the job of the referee to act as time keeper and make any decisions which may need to be made such as fouls, free kicks, throw ins, penalties and added on time at the end of each half. The referee may consult the assistant referees at any time in the match regarding a decision. It’s the assistant referee’s job to spot offside’s in the match (see below), throw ins for either team and also assist the referee in all decision making processes where appropriate.
    •	If the game needs to head to extra time as a result of both teams being level in a match then 30 minutes will be added in the form of two 15 minute halves after the allotted 90 minutes.
    •	If teams are still level after extra time then a penalty shootout must take place.
    •	The whole ball must cross the goal line for it to constitute as a goal.
    •	For fouls committed a player could receive either a yellow or red card depending on the severity of the foul; this comes down to the referee’s discretion. The yellow is a warning and a red card is a dismissal of that player. Two yellow cards will equal one red. Once a player is sent off then they cannot be replaced.
    •	If a ball goes out of play off an opponent in either of the side lines then it is given as a throw in. If it goes out of play off an attacking player on the base line then it is a goal kick. If it comes off a defending player it is a corner kick.""",

    """The Major International Competitions are:
- FIFA World Cup: Unquestionably the biggest sporting event in the world, held every four years. The next edition will be in 2026, hosted by the USA, Canada, and Mexico.
- UEFA European Championship (Euro): Considered the second strongest international tournament due to the high density of elite national teams.
- Copa América: The oldest continental competition in the world, where South American powerhouses like Argentina and Brazil battle for dominance.

The elite club competitions are:
- UEFA Champions League: The pinnacle of club football, featuring the top teams from Europe's strongest leagues. 
The 2025/26 final will take place on May 30, 2026, in Budapest.
- FIFA Club World Cup: In 2025, this tournament expanded to a new 32-team format, putting it on par with major international tournaments.
- Copa Libertadores: The South American equivalent of the Champions League, known for its incredible passion and gritty style of play.

According to UEFA coefficients and market value, the current top leagues are:
1.	Premier League	England	The most financially powerful and most-watched league globally.
2.	Serie A	Italy	A tactically advanced league that has seen a major resurgence recently.
3.	La Liga	Spain	Home to technical football and some of the world's biggest stars.
4.	Bundesliga	Germany	Famous for the highest stadium attendance and attacking play.
5.	Ligue 1	France	A massive source of talent for all other elite leagues.""",

    """This document lists the results of the FIFA World Cup finals from 1930 to 2022, including the winners, runners-up, final scores, host venues and attendances.

- 1930: Uruguay defeated Argentina 4–2 at Estadio Centenario, Uruguay. (Attendance: 68,346)
- 1934: Italy defeated Czechoslovakia 2–1 (after extra time) at Stadio Nazionale PNF, Italy. (Attendance: 55,000)
- 1938: Italy defeated Hungary 4–2 at Stade Olympique de Colombes, France. (Attendance: 45,000)
- 1950: Uruguay defeated Brazil 2–1 at the Maracanã Stadium, Brazil. (Attendance: 173,850)
- 1954: West Germany defeated Hungary 3–2 at Wankdorf Stadium, Switzerland. (Attendance: 62,500)
- 1958: Brazil defeated Sweden 5–2 at Råsunda Stadium, Sweden. (Attendance: 49,737)
- 1962: Brazil defeated Czechoslovakia 3–1 at Estadio Nacional, Chile. (Attendance: 68,679)
- 1966: England defeated West Germany 4–2 (after extra time) at Wembley Stadium, England. (Attendance: 96,924)
- 1970: Brazil defeated Italy 4–1 at Estadio Azteca, Mexico. (Attendance: 107,412)
- 1974: West Germany defeated Netherlands 2–1 at Olympiastadion, West Germany. (Attendance: 78,200)
- 1978: Argentina defeated Netherlands 3–1 (after extra time) at Estadio Monumental, Argentina. (Attendance: 71,483)
- 1982: Italy defeated West Germany 3–1 at Santiago Bernabéu, Spain. (Attendance: 90,000)
- 1986: Argentina defeated West Germany 3–2 at Estadio Azteca, Mexico. (Attendance: 114,600)
- 1990: West Germany defeated Argentina 1–0 at Stadio Olimpico, Italy. (Attendance: 73,603)
- 1994: Brazil defeated Italy 0–0 (3–2 on penalties) at the Rose Bowl, USA. (Attendance: 94,194)
- 1998: France defeated Brazil 3–0 at Stade de France, France. (Attendance: 80,000)
- 2002: Brazil defeated Germany 2–0 at International Stadium, Japan. (Attendance: 69,029)
- 2006: Italy defeated France 1–1 (5–3 on penalties) at Olympiastadion, Germany. (Attendance: 69,000)
- 2010: Spain defeated Netherlands 1–0 (after extra time) at Soccer City, South Africa. (Attendance: 84,490)
- 2014: Germany defeated Argentina 1–0 (after extra time) at Maracanã Stadium, Brazil. (Attendance: 74,738)
- 2018: France defeated Croatia 4–2 at Luzhniki Stadium, Russia. (Attendance: 78,011)
- 2022: Argentina defeated France 3–3 (4–2 on penalties) at Lusail Stadium, Qatar. (Attendance: 88,966)""",

    """Offside can be called when an attacking player is in front of the last defender when the pass is played through to them. The offside area is designed to discourage players from simply hanging around the opponent’s goal waiting for a pass. To be onside they must be placed behind the last defender when the ball is played to them. If the player is in front of that last defender then he is deemed to be offside and free kick to the defending team will be called. A player cannot be caught offside in their own half. The goalkeeper does not count as a defender. If the ball is played backwards and the player is in front of the last defender then he is deemed to be not offside.""",

    """Formation, as most fans know, is the arrangement of players on the field. This is typically expressed as a series of numbers (e.g., 4-4-2, 3-5-2) that indicate the number of players in each horizontal line of the tactical setup. It’s the starting point from which coaches build their game plan.
    
    On the other hand, shape refers to how the team looks during play, which is not always consistent with their formation. It’s about the fluidity of players’ positioning, and is greatly influenced by whether the team has possession, where the ball is located, and their intended strategy at any given moment.""",

    """Often, discussions about formation revolve around how many defenders, midfielders, and forwards are deployed. However, understanding the offensive shape can reveal a side’s true intentions and capability. For instance, a 4-3-3 formation can manifest as a staggered diamond in the midfield when the team is in possession. This allows for a strong pressing game and quick, dynamic wing play. Similarly, a 4-4-2 might shift to a flat four in midfield, offering a more balanced approach between attacking and defending.
    
    A team’s defensive shape is equally illuminating; it’s their first line of resistance against the opponent’s advances. Traditionally, a team might drop into a 4-4-2 or a 5-4-1 when under pressure. However, it’s the intricacies of player positioning within these formations that truly dictate the team’s ability to close down spaces and mark opponents. This involves a mix of zonal marking and man-to-man coverage, all while maintaining the integrity of the defensive shape to avoid becoming disjointed.
    
    One of the most crucial—and least appreciated—aspects of shape is how a team handles the transition from offense to defense and vice versa. This in-between phase is often when teams are most vulnerable, and the successful management of this dynamic can lead to turnovers or prevent counter-attacks.
    
    During an attacking phase, players must anticipate the potential loss of possession and adjust their positioning to make the transition to defense as smooth as possible. Conversely, the defending team must be ready to exploit any lapses as the ball changes hands. A good transition shape ensures that the team remains a cohesive unit, able to adapt swiftly to either mode of play.""",

    """The 4-3-3 is the most widespread system in modern football. Used by teams like Manchester City, Real Madrid, or PSG, it’s known for its balance between defense and attack. The wide wingers provide width, the three midfielders control the game’s rhythm, and the striker is the offensive reference point.
    
    Another key modern formation is the 4-2-3-1, widely used by national teams and top European clubs. It relies on two defensive midfielders who provide stability and a playmaker who operates between the lines, connecting with the striker.
    
    The 3-5-2 has made a comeback in recent years thanks to managers like Antonio Conte and Thomas Tuchel. The key lies in the wing-backs, players who must possess enormous physical and tactical endurance as they cover the entire flank.
    
    Although some see it as a system of the past, the 4-4-2 is still alive, especially in teams that prioritize defensive solidity and structure. With two compact lines and two mobile strikers, it becomes an effective tool for pressing and counterattacking.
    
    Used by national teams like Italy or Croatia, the 5-3-2 is a defensive variation of the 3-5-2. It provides maximum security at the back, with three center-backs and two wing-backs dropping deep to form a line of five.
    
    Finally, the 4-1-4-1 has gained popularity in recent years due to its ability to maintain possession and protect the midfield zone. The defensive midfielder acts as an anchor, while the four midfielders contribute pressure and attacking support.""",

    """WORLD CUP ALL-TIME TOP SCORERS LIST:
1. Miroslav Klose from Germany is the all-time leading goal scorer in World Cup history with 16 goals across 24 appearances.
2. Ronaldo from Brazil is the second highest scorer with 15 goals in 19 appearances.
3. Gerd Müller from West Germany scored 14 goals in only 13 appearances.
4. Lionel Messi from Argentina has scored 13 goals in 26 World Cup appearances.
5. Just Fontaine from France holds the record for most goals in a single tournament, with 13 goals in 6 appearances.
6. Pelé from Brazil scored a total of 12 goals in 14 World Cup appearances.
7. Kylian Mbappé from France has scored 12 goals in 14 appearances so far.
8. Jürgen Klinsmann from Germany scored 11 goals in 17 World Cup matches.
9. Sándor Kocsis from Hungary scored 11 goals in just 5 appearances.
10. Grzegorz Lato from Poland scored 10 goals in 20 appearances.""",

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
def build_vector_store(_documents: tuple    ):
    """Chunk documents, embed them, and store in ChromaDB."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    # --- Chunking ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
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
    page = st.radio("Go to:", ["Home", "Search", "Statistics", "Trivia Quiz", "Tactics Compare"])
    
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


elif page == "Tactics Compare":
    st.title("🛡️ Tactical Comparison Lab")
    st.markdown("Compare modern and classic formations to see how they utilize the pitch.")

    # Dictionary mapping formations to helpful diagrams
    FORMATION_IMAGES = {
        "4-3-3": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Association_football_4-3-3_formation.svg/960px-Association_football_4-3-3_formation.svg.png",
        "4-4-2": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Association_football_4-4-2_formation.svg/960px-Association_football_4-4-2_formation.svg.png",
        "3-5-2": "https://upload.wikimedia.org/wikipedia/commons/a/af/3-5-2DV.jpg",
        "5-3-2": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Association_football_5-3-2_formation.svg/960px-Association_football_5-3-2_formation.svg.png",
        "4-2-3-1": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Association_football_4-2-3-1_formation.svg/960px-Association_football_4-2-3-1_formation.svg.png",
        "4-5-1": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Association_football_4-5-1_formation.svg/960px-Association_football_4-5-1_formation.svg.png",
        "3-4-3": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Soccer_formation_3-4-3.svg/960px-Soccer_formation_3-4-3.svg.png",
        "4-4-2 Diamond": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/Association_football_4-4-2_diamond_formation.svg/960px-Association_football_4-4-2_diamond_formation.svg.png"

    }

    FORMATION_TEXT = {
        "4-3-3": "The 4–3–3 was a development of the 4–2–4, and was played by the Brazil national team in the 1962 World Cup, although a 4–3–3 had also previously been used by the Uruguay national team in the 1950 and 1954 World Cups. The extra player in midfield allows a stronger defence, and the midfield could be staggered for different effects. The three midfielders normally play closely together to protect the defence, and move laterally across the field as a coordinated unit. The formation is usually played without wide midfielders. The three forwards split across the field to spread the attack, and may be expected to mark the opposition full-backs as opposed to doubling back to assist their own full-backs, as do the wide midfielders in a 4–4–2. ",
        "4-4-2": "This formation was the most common in football in the 1990s and early 2000s, in which midfielders are required to work hard to support both the defence and the attack: typically one of the central midfielders is expected to go upfield as often as possible to support the forward pair, while the other will play a holding role, shielding the defence; the two wide midfield players must move up the flanks to the goal line in attacks and yet also protect the full-backs.",
        "4-4-2 Diamond": "The 4–4–2 diamond (also described as 4–1–2–1–2) staggers the midfield. The width in the team has to come from the full-backs pushing forward. The defensive midfielder is sometimes used as a deep-lying playmaker, but needs to remain disciplined and protect the back four behind him. The central attacking midfielder is the creative player, responsible for picking up the ball, and distributing the ball wide to its full-backs or providing the two strikers with through balls. When out of possession, the midfield four must drop and assist the defence, while the two strikers must be free for the counter-attack.",
        "3-5-2": "This formation is similar to 5–3–2, but with some important tweaks: there is usually no sweeper (or libero) but rather three classic centre-backs, and the two wing-backs are oriented more towards the attack. Because of this, the most central midfielder tends to remain further back in order to help prevent counter-attacks. It also differs from the classical 3–5–2 of the WW by having a non-staggered midfield.",
        "3-4-3": "Using a 3–4–3, the midfielders are expected to split their time between attacking and defending. Having only three dedicated defenders means that if the opposing team breaks through the midfield, they will have a greater chance to score than with a more conventional defensive configuration, such as 4–5–1 or 4–4–2. However, the three forwards allow for a greater concentration on attack. This formation is used by more offensive-minded teams.",
        "4-2-3-1": "A flexible formation in prospects to defensive or offensive orientation, as both the wide players and the full-backs may join the attack. In defence, this formation is similar to either the 4–5–1 or 4–4–1–1. It is used to maintain possession of the ball and stop opponent attacks by controlling the midfield area of the field. The lone striker may be very tall and strong to hold the ball up as his midfielders and full-backs join him in attack. The striker could also be very fast. In these cases, the opponent's defence will be forced to fall back early, thereby leaving space for the offensive central midfielder. This formation is used especially when a playmaker is to be highlighted. The variations of personnel used on the flanks in this set-up include using traditional wingers, using inverted wingers or simply using wide midfielders.",
        "4-5-1": "4–5–1 is a conservative formation; however, if the two midfield wingers play a more attacking role, it can be likened to 4–3–3. The formation can be used to grind out 0–0 draws or preserve a lead, as the packing of the centre midfield makes it difficult for the opposition to build up play. Because of the closeness of the midfield, the opposing team's forwards will often be starved of possession. Due to the lone striker, however, the centre of the midfield does have the responsibility of pushing forward as well. The defensive midfielder will often control the pace of the game.",
        "5-3-2": "This formation has three central defenders, possibly with one acting as a sweeper. This system merges the winger and full-back positions into the wing-back, whose job it is to work their flank along the full length of the pitch, supporting both the defence and the attack."
        # If you leave a formation out of this list, the app will use the AI search instead!
    }

    col1, col2 = st.columns(2)
    with col1:
        choice1 = st.selectbox("Select Formation A", list(FORMATION_IMAGES.keys()), key="f1")
        st.image(FORMATION_IMAGES[choice1], caption=f"Standard {choice1} Layout", width=200)
    with col2:
        choice2 = st.selectbox("Select Formation B", list(FORMATION_IMAGES.keys()), key="f2")
        st.image(FORMATION_IMAGES[choice2], caption=f"Standard {choice2} Layout", width=200)

    st.divider()

    if st.button("🔍 Compare Tactical Data", use_container_width=True):
        vector_store, _ = build_vector_store(tuple(DOCUMENTS))
        
        # LOGIC: Check manual text first, then fallback to AI search
        def get_analysis(choice):
            if choice in FORMATION_TEXT:
                return FORMATION_TEXT[choice] # Use your manual text
            else:
                res = vector_store.similarity_search(f"Explain the {choice} formation", k=1)
                return res[0].page_content # Use AI search

        text1 = get_analysis(choice1)
        text2 = get_analysis(choice2)

        analysis_col1, analysis_col2 = st.columns(2)
        with analysis_col1:
            st.markdown(f"### Analysis: {choice1}")
            st.info(text1)
        with analysis_col2:
            st.markdown(f"### Analysis: {choice2}")
            st.success(text2)



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

