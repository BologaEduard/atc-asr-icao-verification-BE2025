import streamlit as st
import auditor_logic as logic
import pandas as pd
import os

# Page Config
st.set_page_config(page_title="ICAO Safety Auditor - Thesis", page_icon="✈️", layout="wide")

# Custom Styling for Aviation Dashboard
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stAlert { border-radius: 10px; }
    .entity-label { color: #8b949e; font-size: 0.8rem; margin-bottom: -5px; }
    .entity-value { font-family: 'Courier New', monospace; font-size: 1.1rem; color: #58a6ff; }
    </style>
""", unsafe_allow_html=True)

# Load Resources (Initializing both ICAO databases and the ASR Pipeline)
if 'init' not in st.session_state:
    with st.spinner("Loading ICAO Databases & ATC Speech Models..."):
        st.session_state.phr, st.session_state.embeddings = logic.init_data()
        st.session_state.client = logic.get_openai_client()
        st.session_state.asr_pipeline = logic.init_asr() # Initialize Wav2Vec2 Pipeline
        st.session_state.init = True

# Header
st.title("✈️ ICAO Phraseology Verification Report")
st.markdown("Automated Safety Auditor for ICAO Doc 4444 Compliance")
st.divider()

# --- INPUT METHOD SELECTION ---
st.subheader("📥 Input Method")
input_mode = st.radio("Select input source:", ["Type Transcript", "Upload ATC Audio"], horizontal=True)

final_transcript = ""

if input_mode == "Type Transcript":
    final_transcript = st.text_input("🎧 UTTERANCE TRANSCRIPT:", 
                                     placeholder="Enter transmission (e.g., Malev 123 climb flight level 250)")
else:
    uploaded_file = st.file_uploader("Upload ATC Audio File (WAV/MP3):", type=["wav", "mp3"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        if st.button("GENERATE TRANSCRIPT FROM AUDIO"):
            with st.spinner("AI is transcribing ATC speech..."):
                # Save temp file for the pipeline processing
                temp_path = "temp_upload.wav"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Transcribe using logic built from the Jzuluaga model
                final_transcript = logic.transcribe_audio(st.session_state.asr_pipeline, temp_path)
                st.session_state.current_transcript = final_transcript
                os.remove(temp_path)
    
    # Use session state to persist transcribed text for analysis
    if 'current_transcript' in st.session_state:
        final_transcript = st.text_area("REVIEW TRANSCRIPT:", value=st.session_state.current_transcript)

# --- ANALYSIS TRIGGER ---
if st.button("RUN COMPLIANCE & SAFETY AUDIT", use_container_width=True) and final_transcript:
    # Stage 1: Local Hybrid Filter (Semantic + Fuzzy)
    with st.spinner("Searching ICAO database..."):
        candidates = logic.local_filter_candidates(final_transcript, st.session_state.phr, st.session_state.embeddings)
    
    # Stage 2: LLM Contextual Reasoning based on the Extended System Prompt
    with st.spinner("Performing Safety Audit..."):
        report = logic.llm_verify(st.session_state.client, final_transcript, candidates)

    # --- UI REPRESENTATION ---
    risk_score = report.get("operational_safety_risk_score", 0.0)
    risk_cat = report.get("risk_category", "UNKNOWN")

    # 1. High-Level Metrics Row (Compliance, Accuracy, Risk)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("ICAO Compliance", f"{report.get('icao_compliance_score', 0):.2f}")
    with m2:
        st.metric("Semantic Accuracy", f"{report.get('domain_specific_accuracy_score', 0):.2f}")
    with m3:
        st.metric("Safety Risk", risk_cat, delta=f"{risk_score:.2f}", delta_color="inverse")

    # 2. Safety Status Banner (Logic from notebook safety check)
    if risk_score >= 0.8:
        st.error(f"🚨 **{risk_cat} RISK:** {report.get('reasoning_summary')}")
    elif risk_score >= 0.4:
        st.warning(f"⚠️ **{risk_cat} RISK:** {report.get('reasoning_summary')}")
    else:
        st.success(f"✅ **{risk_cat} RISK:** {report.get('reasoning_summary')}")

    # 3. Main Analysis Layout
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📌 Best ICAO Match")
        best = report.get("best_match", {})
        st.markdown(f"""
        **ID:** `{best.get('id', 'N/A')}` | **Section:** {best.get('section', 'N/A')}  
        **Standard Phraseology:** > `{best.get('icao_phrase', 'N/A')}`
        """)

    with col_right:
        st.subheader("🆔 Core Entities (NER)")
        entities = report.get("core_entity_extraction", {})
        for key, value in entities.items():
            st.markdown(f"<p class='entity-label'>{key.replace('_', ' ').upper()}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='entity-value'>{value}</p>", unsafe_allow_html=True)

    st.divider()

    # 4. Mandatory Elements & Required Improvements
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🛑 Missing Elements")
        missing = report.get("missing_information", [])
        if missing:
            for item in missing:
                st.write(f"❌ {item}")
        else:
            st.write("✅ No missing mandatory elements detected.")

    with c2:
        st.subheader("✏️ Required Corrections")
        corrections = report.get("corrections_needed", [])
        if corrections:
            for item in corrections:
                st.write(f"🔸 {item}")
        else:
            st.write("✅ No corrections needed.")