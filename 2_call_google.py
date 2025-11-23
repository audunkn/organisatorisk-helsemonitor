import os
import glob
import pandas as pd
import google.generativeai as genai
import time
import hashlib 
from dotenv import load_dotenv
import mlflow 
from mlflow import log_metric, log_param, log_artifact

# --- KONFIGURASJON ---
MLFLOW_EXPERIMENT_NAME = "Business Stability Analysis - 25 calls"

load_dotenv() 

API_KEY = os.getenv("GEMINI_API_KEY") 
if not API_KEY:
    raise ValueError("GEMINI_API_KEY er ikke funnet.")

genai.configure(api_key=API_KEY)

MODEL_NAME = 'models/gemini-2.5-flash-preview-09-2025'
PROMPT_DIR = "prompts" 

# Liste over driverkolonner (Brukes for beregning og logging)
DRIVER_COLUMNS = [
    "Macro_Env", "Supply_Chain", "Manufacturing_Quality", 
    "Human_Capital", "Demand_Patterns", "Pricing_Power", "Strategic_Execution"
]

# --- Hjelpefunksjoner ---

def load_prompt(filename):
    """Henter promptteksten fra den dedikerte mappen."""
    filepath = os.path.join(PROMPT_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Feil: Prompt-filen '{filepath}' ble ikke funnet.")
        raise

def get_stability_score(transcript_text):
    """
    Kjører analysen ved å bruke prompt fra business_stability_prompt.txt
    
    KORRIGERT RENSING FOR Å HÅNDTERE MARKDOWN!
    """
    model = genai.GenerativeModel(MODEL_NAME)
    prompt_template = load_prompt('business_stability_prompt.txt')
    prompt = prompt_template.format(transcript_text=transcript_text) 
    
    try:
        response = model.generate_content(prompt)
        # --- NY RENSING: Fjerner * (Markdown bold) og punktum . ---
        clean_score = response.text.strip().replace('+', '').replace('*', '').replace('.', '')
        # -----------------------------------------------------------------
        return int(clean_score)
    except Exception as e:
        print(f"Feil under stabilitets-score: {e}")
        return 0

def get_driver_analysis(transcript_text, stability_score):
    """
    Kjører analysen ved å bruke prompt fra driver_analysis_prompt.txt
    """
    model = genai.GenerativeModel(MODEL_NAME)
    prompt_template = load_prompt('driver_analysis_prompt.txt')
    prompt = prompt_template.format(stability_score=stability_score, transcript_text=transcript_text)
    
    try:
        response = model.generate_content(prompt)
        # Rensing av tekst for å sikre at vi kun får tallene
        # Merk: Her må koden håndtere en streng som "-2, -1, 0, ..."
        # Vi antar at AI følger instruksjonen om rent format.
        return response.text.strip()
    except Exception as e:
        print(f"Feil under driver-analyse: {e}")
        return "0,0,0,0,0,0,0"


# --- HOVEDFUNKSJON MED MLFLOW IMPLEMENTERING (Uendret i logikk) ---

def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        
        print("MLflow Run startet.")
        
        # --- 1. Logg Prompts og Parametere ---
        stability_prompt = load_prompt('business_stability_prompt.txt')
        driver_prompt = load_prompt('driver_analysis_prompt.txt')

        log_param("model_name", MODEL_NAME)
        log_param("stability_prompt_hash", hashlib.sha256(stability_prompt.encode()).hexdigest())
        log_param("driver_prompt_hash", hashlib.sha256(driver_prompt.encode()).hexdigest())
        
        with open(os.path.join(PROMPT_DIR, 'business_stability_prompt.txt'), 'r') as f:
             log_artifact(os.path.join(PROMPT_DIR, 'business_stability_prompt.txt'), artifact_path="prompts")
        with open(os.path.join(PROMPT_DIR, 'driver_analysis_prompt.txt'), 'r') as f:
             log_artifact(os.path.join(PROMPT_DIR, 'driver_analysis_prompt.txt'), artifact_path="prompts")


        # --- 2. Kjør Analyse ---

        transcript_files = glob.glob("full_transcripts_output/*.txt")
        if not transcript_files:
            print("Fant ingen .txt filer.")
            return

        results = []
        print(f"Starter analyse av {len(transcript_files)} filer...")

        for filename in transcript_files:
            
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                
            stability_score = get_stability_score(content)
            driver_raw = get_driver_analysis(content, stability_score)
            
            try:
                driver_scores = [int(x.strip()) for x in driver_raw.split(',')]
                if len(driver_scores) != 7:
                    driver_scores = driver_scores + [0]*(7-len(driver_scores))
            except:
                driver_scores = [0, 0, 0, 0, 0, 0, 0]

            row = {
                "Filnavn": filename,
                "Macro_Env": driver_scores[0],
                "Supply_Chain": driver_scores[1],
                "Manufacturing_Quality": driver_scores[2],
                "Human_Capital": driver_scores[3],
                "Demand_Patterns": driver_scores[4],
                "Pricing_Power": driver_scores[5],
                "Strategic_Execution": driver_scores[6],
                "Overall_Stability_Score": stability_score
            }
            results.append(row)
            time.sleep(2) 

        # --- 3. Lagre og Logg Resultater ---
        df = pd.DataFrame(results)
        output_filename = "analyse_resultater.csv"
        df.to_csv(output_filename, index=False, sep=';') 

        # Logg hovedmetrikker
        avg_score = df["Overall_Stability_Score"].mean()
        log_metric("average_stability_score", avg_score)
        log_metric("total_runs", len(df))
        
        # Logg gjennomsnittsscore for HVER DRIVER
        driver_means = df[DRIVER_COLUMNS].mean()
        print("\nDriver Gjennomsnitt:")
        for driver in DRIVER_COLUMNS:
            metric_name = f"avg_{driver}_score"
            score = driver_means[driver]
            log_metric(metric_name, score)
            print(f"  {metric_name}: {score:.2f}")


        # Logg den genererte CSV-filen som en MLflow artefakt
        log_artifact(output_filename) 
        
        print(f"\nFerdig! MLflow Run fullført. Gjennomsnittlig totalscore: {avg_score:.2f}")

if __name__ == "__main__":
    main()