# atc-asr-icao-verification-BE2025
# ATC Speech Recognition and ICAO Phraseology Verification

This repository contains the software implementation for my thesis focusing on Air Traffic Control (ATC) radio communication processing. The project integrates state-of-the-art ASR models with a custom LLM-based verification layer to ensure safety and ICAO Doc 4444 compliance.

## Key Features
- **Acoustic Robustness Testing:** Python scripts for Additive White Gaussian Noise (AWGN) injection and RMS-based SNR calculation to simulate real-world VHF radio conditions.
- **ASR Evaluation:** Comparative analysis of Whisper-large-v3 (fine-tuned) and Wav2Vec2-ATC models using WER and CER metrics.
- **ICAO Verification:** A two-stage verification system using fuzzy matching and GPT-4o-mini to validate ATC phraseology, detect dangerous contradictions, and extract core entities (NER).

## Repository Structure
- `/notebooks`: Data normalization, training logs, and evaluation metrics.
- `/scripts`: Audio processing, noise simulation, and LLM API integration.
- `/data_samples`: Example input/output structures (excluding private datasets).

*Note: This repository is part of a BME VIK Bachelor's Thesis.*
