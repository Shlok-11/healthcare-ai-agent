import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Page Setup
st.set_page_config(page_title="Healthcare AI Agent", page_icon="🏥", layout="wide")
st.title("🏥 Healthcare AI Assistant")
st.markdown("Ask me anything about the State of AI in Healthcare (2025-2026).")

DB_DIR = os.path.join("data", "chroma_db")

@st.cache_resource
def load_database():
    """Agent 1's Memory: Loads the local Chroma database"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vector_db

@st.cache_resource
def load_llm():
    """Agent 2's Brain: Loads the Llama 3.3 70B model via Groq"""
    # ⚠️ WARNING: Rotate this API key as soon as possible since it was shared.
    my_groq_key = st.secrets["GROQ_API_KEY"] 
    
    return ChatGroq(
        api_key=my_groq_key,
        temperature=0.0,  # Hyper-focused
        max_tokens=300,   # Hard limit to prevent looping
        model_name="llama-3.3-70b-versatile" # <-- MASSIVE 70B UPGRADE
    )

db = load_database()
llm = load_llm()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("E.g., What is the adoption rate of Ambient Notes?"):
    
    # Save user message to UI state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- FORMAT CHAT HISTORY FOR AGENTS ---
    chat_history = ""
    for msg in st.session_state.messages[:-1]: 
        chat_history += f"{msg['role'].capitalize()}: {msg['content']}\n"

    # Multi-Agent Processing Pipeline
    with st.chat_message("assistant"):
        
        # --- AGENT 1: DECOMPOSER & RETRIEVER ---
        st.markdown("### 🔍 Agent 1: Query Decomposition & Retrieval")
        
        decompose_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strict search query generator for a HEALTHCARE AI database. 
            CRITICAL RULES:
            1. Read the Chat History to resolve any pronouns.
            2. Break the question into EXACTLY 3 distinct, keyword-rich search queries. 
            3. If the question asks for multiple things (e.g., a drug, a disease), map one query to each concept.
            4. DOMAIN LOCK: Assume ALL questions are about Healthcare AI, clinical tools, or hospitals.
            5. OUTPUT FORMAT: Just the 3 queries on separate lines. NO EXPLANATIONS. NO PREAMBLE. NO BULLET POINTS.
            6. DATE FLEXIBILITY: If the user asks for a specific year (like 2025), omit the exact year from at least one of your search queries so you don't miss historical data.
            
            EXAMPLE:
            History: None
            Question: What Phase II milestone did Insilico reach, what disease does it target, and what company merger reshaped the landscape?
            Output:
            Insilico Medicine ISM001-055 Rentosertib Phase II milestone
            Insilico Rentosertib target disease
            AI drug discovery company merger landscape"""),
            ("human", "Chat History:\n{history}\n\nLatest Question: {question}")
        ])
        
        decompose_chain = decompose_prompt | llm
        sub_queries_response = decompose_chain.invoke({
            "history": chat_history, 
            "question": prompt
        }).content
        
        sub_queries = [q.strip() for q in sub_queries_response.split('\n') if q.strip()]
        
        st.write("**Decomposed into Sub-queries:**")
        for sq in sub_queries:
            st.write(f"- `{sq}`")
            
        # Step 1B: Retrieve docs using MMR for Maximum Diversity
        all_results = []
        for sq in sub_queries:
            # MMR looks at 15 chunks, but only returns the 4 most DIVERSE ones
            all_results.extend(db.max_marginal_relevance_search(sq, k=4, fetch_k=15))
            
        # Step 1C: Remove duplicate docs
        unique_docs = {}
        for doc in all_results:
            unique_docs[doc.page_content] = doc
        
        # Step 1D: Assemble the context package and display as INDIVIDUAL CHUNKS
        context_text = ""
        for i, doc in enumerate(list(unique_docs.values())[:15]): # Increased to 12 safe chunks
            source_id = doc.metadata.get('source', 'Unknown').split('\\')[-1].replace('.txt', '')
            
            # Display each chunk in its own UI dropdown
            with st.expander(f"📄 Retrieved Chunk {i+1} ({source_id})"):
                st.write(doc.page_content)
            
            # Stack the text for Agent 2 to read
            context_text += f"\n[Source: {source_id}]\nContent: {doc.page_content}\n"

        # --- AGENT 2: REASONER & SYNTHESIZER ---
        st.markdown("### 🧠 Agent 2: Chain of Thought")
        
        system_prompt = """You are an intelligent Healthcare AI assistant. 
        Your goal is to be helpful and provide the best possible answer using the Context Package.
        
        GUIDELINES:
        1. EVIDENCE-BASED SYNTHESIS: Base your answers on the provided documents. You are expected to use deductive reasoning, connect related concepts, and recognize industry synonyms (e.g., "Ambient Notes" and "Clinical Documentation" mean the same thing).
        2. DO NOT INVENT DATA: While you can be flexible with language and logic, never make up statistics, percentages, or specific medical claims. 
        3. PARTIAL ANSWERS ARE GREAT: If a user asks a complex question and you only find part of the answer in the text, provide what you have! Just briefly mention what is missing.
        4. CITATIONS: End your factual claims with the source format: (Art_XX).
        
        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        <thinking>
        [Briefly explain your logic, how you connected the concepts, or note any missing details. Max 3 sentences.]
        </thinking>
        Final Answer: [Your concise, cited answer here]
        
        Chat History:
        {history}
        
        Context Package:
        {context}
        """
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        chain = prompt_template | llm
        
        # Stream the response
        response_placeholder = st.empty()
        full_response = ""
        
        for chunk in chain.stream({
            "context": context_text, 
            "history": chat_history, 
            "question": prompt
        }):
            full_response += chunk.content
            response_placeholder.markdown(full_response + "▌")
            
        response_placeholder.markdown(full_response)
        
    # Save the assistant's final response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
