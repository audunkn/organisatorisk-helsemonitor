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
# Endret navn til norsk. Dette blir "mappen" i MLflow.
MLFLOW_EXPERIMENT_NAME = "Analyse av Forretningsstabilitet"

load_dotenv() 

API_KEY = os.getenv("GEMINI_API_KEY") 
if not API_KEY:
    raise ValueError("GEMINI_API_KEY er ikke funnet.")

genai.configure(api_key=API_KEY)

MODEL_NAME = 'models/gemini-2.5-flash-preview-09-2025'
PROMPT_DIR = "prompts" 

# Liste over driverkolonner (Norske navn)
# Rekkefølgen MÅ matche output fra den engelske prompten (0-6)
DRIVER_COLUMNS = [
    "Makroforhold", 
    "Forsyningskjede", 
    "Produksjonskvalitet", 
    "Kompetanse", 
    "Etterspørselsmønstre", 
    "Prismakt", 
    "Strategigjennomføring"
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
    Output er et tall (Business Stability Score).
    """
    model = genai.GenerativeModel(MODEL_NAME)
    prompt_template = load_prompt('business_stability_prompt.txt')
    prompt = prompt_template.format(transcript_text=transcript_text) 
    
    try:
        response = model.generate_content(prompt)
        # Renser vekk Markdown-formatering (*, +) og punktum
        clean_score = response.text.strip().replace('+', '').replace('*', '').replace('.', '')
        return int(clean_score)
    except Exception as e:
        print(f"Feil under stabilitets-score: {e}")
        return 0

def get_driver_analysis(transcript_text, stability_score):
    """
    Kjører analysen ved å bruke prompt fra driver_analysis_prompt.txt
    Output er en streng med tall separert av komma.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    prompt_template = load_prompt('driver_analysis_prompt.txt')
    prompt = prompt_template.format(stability_score=stability_score, transcript_text=transcript_text)
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Feil under driver-analyse: {e}")
        return "0,0,0,0,0,0,0"


# --- HOVEDFUNKSJON MED MLFLOW IMPLEMENTERING ---

def main():
    # Setter eksperimentet (lager nytt hvis det ikke finnes)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        
        print(f"MLflow Run startet i eksperimentet: '{MLFLOW_EXPERIMENT_NAME}'")
        
        # --- 1. Logg Prompts og Parametere ---
        stability_prompt = load_prompt('business_stability_prompt.txt')
        driver_prompt = load_prompt('driver_analysis_prompt.txt')

        log_param("model_name", MODEL_NAME)
        # Hasher promptene for versjonskontroll i MLflow
        log_param("stability_prompt_hash", hashlib.sha256(stability_prompt.encode()).hexdigest())
        log_param("driver_prompt_hash", hashlib.sha256(driver_prompt.encode()).hexdigest())
        
        # Lagrer selve prompt-filene
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
            
            # Henter score og driver-tall
            stability_score = get_stability_score(content)
            driver_raw = get_driver_analysis(content, stability_score)
            
            try:
                # Konverterer CSV-streng til liste med heltall
                driver_scores = [int(x.strip()) for x in driver_raw.split(',')]
                
                if len(driver_scores) != 7:
                    driver_scores = driver_scores + [0]*(7-len(driver_scores))
            except:
                driver_scores = [0, 0, 0, 0, 0, 0, 0]

            # --- BYGGER DATARADEN (NORSKE FELTNAVN) ---
            row = {
                "Filnavn": filename,
                "Makroforhold": driver_scores[0],
                "Forsyningskjede": driver_scores[1],
                "Produksjonskvalitet": driver_scores[2],
                "Kompetanse": driver_scores[3],
                "Etterspørselsmønstre": driver_scores[4],
                "Prismakt": driver_scores[5],
                "Strategigjennomføring": driver_scores[6],
                # Endret fra Overall_Stability_Score til Forretningsstabilitet
                "Forretningsstabilitet": stability_score 
            }
            results.append(row)
            time.sleep(1.5) # Litt pause for API-rate limit

        # --- 3. Lagre og Logg Resultater ---
        df = pd.DataFrame(results)
        output_filename = "analyse_resultater.csv"
        df.to_csv(output_filename, index=False, sep=';') 

        # --- LOGGING AV METRIKKER (REN NORSK, INGEN "AVG") ---
        
        # Hovedscore
        avg_stability = df["Forretningsstabilitet"].mean()
        # Her logger vi bare "Forretningsstabilitet". I konteksten av et "Run" er dette snittet.
        log_metric("Forretningsstabilitet", avg_stability)
        log_metric("Antall_analysert", len(df))
        
        # Driverscorer
        driver_means = df[DRIVER_COLUMNS].mean()
        print("\nResultater (Gjennomsnitt):")
        print(f"  Hovedscore (Forretningsstabilitet): {avg_stability:.2f}")
        
        for driver in DRIVER_COLUMNS:
            score = driver_means[driver]
            # Logger navnet direkte (f.eks. "Kompetanse") uten prefix/suffix
            log_metric(driver, score) 
            print(f"  {driver}: {score:.2f}")

        # Laster opp CSV-filen til MLflow
        log_artifact(output_filename) 
        
        print(f"\nFerdig! Resultater lagret i {output_filename}")

if __name__ == "__main__":
    main()