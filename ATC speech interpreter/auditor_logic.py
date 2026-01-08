import pandas as pd
import re
import difflib
import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

# Load model once at startup
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def init_data():
    phr = pd.read_csv("phrases_enriched.csv")
    icao_descriptions = phr["Phraseology"].astype(str).tolist()
    icao_embeddings = embedder.encode(icao_descriptions, convert_to_tensor=True)
    return phr, icao_embeddings

def get_openai_client():
    try:
        with open("API_KEY.txt") as f:
            api_key = f.read().strip()
        return OpenAI(api_key=api_key)
    except:
        return None

def normalize(t):
    t = str(t).upper()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def extract_numeric_slots(text):
    """Exactly from Block 5 of your notebook."""
    text = text.upper()
    return {
        "flight_levels": re.findall(r"FL\s*\d+|FLIGHT LEVEL\s*\d+", text),
        "numbers": re.findall(r"\d+", text),
        "callsigns": re.findall(r"\b[A-Z]{2,3}\d{1,4}[A-Z]?\b", text)
    }

def icao_structure_score(user_norm, row):
    score = 0.0
    phrase = normalize(row["Phraseology"])
    slots = row.get("Slots", "")
    if phrase.split(" ")[0] == user_norm.split(" ")[0]: score += 0.25
    if isinstance(slots, str):
        if "level" in slots.lower() and re.search(r"\bFLIGHT LEVEL\b|\bLEVEL\b|\bFL\s*\d+", user_norm):
            score += 0.15
        if "number" in slots.lower() and re.search(r"\b\d+\b", user_norm):
            score += 0.10
    return score

def sid_star_penalty(user_norm, phrase_norm):
    if (" SID " in f" {phrase_norm} " or " STAR " in f" {phrase_norm} ") and \
       not (" SID " in f" {user_norm} " or " STAR " in f" {user_norm} "):
        return -0.40
    return 0.0

def local_filter_candidates(user_text, phr, icao_embeddings, top_n=5):
    user_norm = normalize(user_text)
    user_embedding = embedder.encode(user_norm, convert_to_tensor=True)
    cos_scores = util.cos_sim(user_embedding, icao_embeddings)[0]
    results = []
    for idx, row in phr.iterrows():
        semantic_sim = cos_scores[idx].item()
        phrase_norm = normalize(row["Phraseology"])
        fuzzy_sim = difflib.SequenceMatcher(None, user_norm, phrase_norm).ratio()
        combined_sim = (0.70 * semantic_sim) + (0.30 * fuzzy_sim)
        final_score = (0.75 * combined_sim + 0.20 * (icao_structure_score(user_norm, row)) + sid_star_penalty(user_norm, phrase_norm))
        results.append((final_score, row))
    results.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in results[:top_n]]

def llm_verify(client, user_text, candidates):
    """
    Stage 2: Sends the user utterance and top candidates to the LLM for deep, 
    context-aware verification, returning a structured JSON report.
    """
    if client is None:
        return {
            "best_match": {"id": "N/A", "section": "N/A", "section_title": "N/A", "icao_phrase": "N/A", "similarity": 0.0},
            "missing_information": ["LLM Client is not initialized (API Key missing)"],
            "corrections_needed": [],
            "icao_compliance_score": 0.0,
            "domain_specific_accuracy_score": 0.0,
            "operational_safety_risk_score": 1.0,
            "risk_category": "CRITICAL",
            "core_entity_extraction": {"command": "N/A", "value": "N/A", "call_sign": "N/A"},
            "reasoning_summary": "LLM client failed to initialize due to missing API key."
        }

    # 1. Prepare candidate text including Sender/Receiver context as per original code
    cand_text = "\n".join([
        f"- {row['ID_str']} | {row['Section']} | {row['Phraseology']} | Sender: {row['Sender']} -> {row['Receiver']}"
        for row in candidates
    ])

    # 2. Extract deterministic slots for the LLM to cross-check (Block 5 logic)
    slots = extract_numeric_slots(user_text)

    # 3. Your Extended System Prompt (Block 6)
    system_prompt = """
You are an expert ICAO Radiotelephony Safety Auditor (Doc 4444).
Your task is to analyze the USER UTTERANCE against the provided CANDIDATE ICAO PHRASES and provide a structured safety and compliance report.

### SCORING DEFINITIONS:
1. 'icao_compliance_score' (0.0 - 1.0): Measures strict adherence to ICAO standard phraseology.
2. 'domain_specific_accuracy_score' (0.0 - 1.0): Measures semantic fidelity. Did the pilot/ATC understand the core intent and value, even if the phrasing was non-standard?
3. 'operational_safety_risk_score' (0.0 - 1.0): 
   - 0.0-0.3: Low risk (minor phrasing errors).
   - 0.4-0.6: Medium risk (missing callsign or slightly ambiguous instruction).
   - 0.7-1.0: Critical risk (action/value contradiction, e.g., 'Climb' to a lower level, or numerical readback mismatch).

### CRITICAL INSTRUCTIONS:
- CROSS-CHECK: Use the 'DETERMINISTIC SLOTS' (provided in the user prompt) to ensure the 'core_entity_extraction' is accurate. Do not hallucinate numbers.
- CALLSIGN CONSISTENCY: If you extract a callsign in 'core_entity_extraction', do not mark it as 'Missing' in the 'missing_information' field unless the callsign itself is incomplete or improper.
- ACTION VALIDATION: Check if the Command (e.g., CLIMB) matches the Value (e.g., a higher FL). If they contradict, the risk score MUST be > 0.8.

### JSON OUTPUT FORMAT (STRICT):
{
  "best_match": {
    "id": "String (e.g., 12.3.1.2)",
    "section": "String",
    "section_title": "String",
    "icao_phrase": "String",
    "similarity": 0.0-1.0
  },
  "missing_information": ["List", "of", "missing", "mandatory", "elements"],
  "corrections_needed": ["List", "of", "suggested", "wording", "improvements"],
  "icao_compliance_score": 0.0-1.0,
  "domain_specific_accuracy_score": 0.0-1.0,
  "operational_safety_risk_score": 0.0-1.0,
  "risk_category": "LOW | MEDIUM | HIGH | CRITICAL",
  "core_entity_extraction": {
    "command": "Primary action (e.g., CLIMB, MAINTAIN)",
    "value": "Numerical value or point (e.g., FL200)",
    "call_sign": "Aircraft identification or 'NONE'"
  },
  "reasoning_summary": "Concise professional explanation of the safety and compliance status."
}

Return ONLY the JSON object.
"""

    user_prompt = f"""
USER UTTERANCE: "{user_text}"
DETERMINISTIC SLOTS DETECTED: {slots}

CANDIDATE ICAO PHRASES:
{cand_text}

Compare the slots in the utterance with the expected slots in the candidates.
Return JSON ONLY.
"""

    # 4. Modern OpenAI API call with JSON mode
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )

    # Return as a dictionary so app.py can use it immediately
    return json.loads(response.choices[0].message.content)

import librosa
import torch
import os
from transformers import pipeline

# --- Speeech Recognition Logic ---

import librosa
import torch
from transformers import pipeline

def init_asr():
    """Initializes ASR for CPU. Handles the missing KenLM gracefully."""
    MODEL_ID = "Jzuluaga/wav2vec2-xls-r-300m-en-atc-uwb-atcc"
    
    # Force CPU
    device = -1 
    
    # Initialize pipeline
    # It will automatically fall back to 'Raw CTC' since KenLM isn't found
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        device=device
    )
    return asr_pipeline

def transcribe_audio(asr_pipeline, audio_path):
    """
    CPU Optimization: Resample to 16kHz before processing.
    This prevents the CPU from hanging on high-bitrate files.
    """
    # Load and force 16kHz (Standard for Wav2Vec2)
    speech, _ = librosa.load(audio_path, sr=16000)
    
    # Run transcription
    result = asr_pipeline(speech)
    
    # Clean up output
    return result["text"].lower()