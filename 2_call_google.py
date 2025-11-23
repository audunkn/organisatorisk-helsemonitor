import os
import glob
import pandas as pd
import google.generativeai as genai
import time
import hashlib 
import re  # <--- NY: Vi trenger denne for å finne tall i tekst
from dotenv import load_dotenv
import mlflow 
from mlflow import log_metric, log_param, log_artifact

# --- KONFIGURASJON ---
MLFLOW_EXPERIMENT_NAME = "Analyse av Forretningsstabilitet - v1"

load_dotenv() 

API_KEY = os.getenv("GEMINI_API_KEY") 
if not API_KEY:
    raise ValueError("GEMINI_API_KEY er ikke funnet.")

genai.configure(api_key=API_KEY)

MODEL_NAME = 'models/gemini-2.5-flash-preview-09-2025'
PROMPT_DIR = "prompts" 

DRIVER_COLUMNS = [
    "Makroforhold", "Forsyningskjede", "Produksjonskvalitet", 
    "Kompetanse", "Etterspørselsmønstre", "Prismakt", "Strategigjennomføring"
]

# --- Hjelpefunksjoner ---

def load_prompt(filename):
    filepath = os.path.join(PROMPT_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def get_stability_score(transcript_text):
    """Henter Business Stability Score og bruker Regex for å finne tallet."""
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = load_prompt('business_stability_prompt.txt').format(transcript_text=transcript_text) 
    
    try:
        response = model.generate_content(prompt)
        # --- NY ROBUST METODE ---
        # Finner alle tall (inkludert negative) i teksten
        matches = re.findall(r'-?\d+', response.text)
        if matches:
            return int(matches[0]) # Returnerer det første tallet den finner
        return 0
    except Exception as e:
        print(f"Feil under stabilitets-score: {e}")
        return 0

def get_driver_analysis(transcript_text, stability_score):
    """Henter driver-scorene."""
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = load_prompt('driver_analysis_prompt.txt').format(
        stability_score=stability_score, 
        transcript_text=transcript_text
    )
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Feil under driver-analyse: {e}")
        return "" # Returnerer tom streng ved feil

# --- HOVEDFUNKSJON ---

def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        print(f"MLflow Run startet: {MLFLOW_EXPERIMENT_NAME}")
        
        # Logg prompts
        stability_prompt = load_prompt('business_stability_prompt.txt')
        driver_prompt = load_prompt('driver_analysis_prompt.txt')
        log_param("model_name", MODEL_NAME)
        
        transcript_files = glob.glob("full_transcripts_output/*.txt")
        print(f"Analyserer {len(transcript_files)} filer...")

        results = []

        for filename in transcript_files:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 1. Hent Hovedscore
            stability_score = get_stability_score(content)
            
            # 2. Hent Drivere
            driver_raw = get_driver_analysis(content, stability_score)
            
            # --- NY ROBUST PARSING AV LISTEN ---
            try:
                # Regex finner alle heltall (f.eks -2, 5, 0) i strengen, uavhengig av tekst rundt
                numbers = re.findall(r'-?\d+', driver_raw)
                
                # Konverter til int
                driver_scores = [int(n) for n in numbers]
                
                # Sørg for at vi har nøyaktig 7 tall. 
                # Hvis vi fant for få, fyller vi på med 0. Hvis for mange, kutter vi.
                if len(driver_scores) < 7:
                    driver_scores += [0] * (7 - len(driver_scores))
                else:
                    driver_scores = driver_scores[:7]
                    
            except Exception as e:
                print(f"Kunne ikke parse tall fra: '{driver_raw}'. Feil: {e}")
                driver_scores = [0] * 7

            print(f"Fil: {os.path.basename(filename)} -> Score: {stability_score}, Drivere: {driver_scores}")

            # Bygg datarad
            row = {
                "Filnavn": filename,
                "Makroforhold": driver_scores[0],
                "Forsyningskjede": driver_scores[1],
                "Produksjonskvalitet": driver_scores[2],
                "Kompetanse": driver_scores[3],
                "Etterspørselsmønstre": driver_scores[4],
                "Prismakt": driver_scores[5],
                "Strategigjennomføring": driver_scores[6],
                "Forretningsstabilitet": stability_score 
            }
            results.append(row)
            time.sleep(1) # Skånsom mot API-et

        # Lagre
        df = pd.DataFrame(results)
        output_filename = "analyse_resultater.csv"
        df.to_csv(output_filename, index=False, sep=';') 
        
        # Logg til MLflow
        avg_stability = df["Forretningsstabilitet"].mean()
        log_metric("Forretningsstabilitet", avg_stability)
        log_metric("Antall_analysert", len(df))
        log_artifact(output_filename) 
        
        print(f"\nFerdig! Resultater lagret i {output_filename}")

if __name__ == "__main__":
    main()