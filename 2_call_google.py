import os
import glob
import pandas as pd
import google.generativeai as genai
import time
import hashlib 
import re 
from dotenv import load_dotenv
import mlflow 
from mlflow import log_metric, log_param, log_artifact

# --- KONFIGURASJON ---
MLFLOW_EXPERIMENT_NAME = "Analyse av Forretningsstabilitet - v2"

load_dotenv() 

API_KEY = os.getenv("GEMINI_API_KEY") 
if not API_KEY:
    raise ValueError("GEMINI_API_KEY er ikke funnet.")

genai.configure(api_key=API_KEY)

MODEL_NAME = 'models/gemini-2.5-flash-preview-09-2025'
PROMPT_DIR = "prompts" 

# --- Hjelpefunksjoner ---

def load_prompt(filename):
    filepath = os.path.join(PROMPT_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def generate_content_with_retry(model, prompt, max_retries=5, initial_wait=30):
    """
    Prøver å kalle Gemini API-et. Hvis vi treffer Rate Limit (429),
    venter vi og prøver igjen.
    """
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                wait_time = initial_wait * (attempt + 1) # Øker ventetiden for hvert forsøk
                print(f"⚠️ Traff Rate Limit (429). Venter {wait_time} sekunder før nytt forsøk ({attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                # Hvis det er en annen feil enn Rate Limit, kast den videre (eller print og gi opp)
                print(f"❌ Uventet feil fra API: {e}")
                return None
    
    print("❌ Ga opp etter maksimale gjentakelser.")
    return None

def get_stability_score(transcript_text):
    """Henter Business Stability Score med Retry-logikk."""
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = load_prompt('business_stability_prompt.txt').format(transcript_text=transcript_text) 
    
    # Bruker retry-funksjonen
    response = generate_content_with_retry(model, prompt)
    
    if response and response.text:
        try:
            matches = re.findall(r'-?\d+', response.text)
            if matches:
                return int(matches[0])
        except Exception as e:
            print(f"Feil ved parsing av score: {e}")
            
    return 0

def get_driver_analysis(transcript_text, stability_score):
    """Henter driver-scorene med Retry-logikk."""
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = load_prompt('driver_analysis_prompt.txt').format(
        stability_score=stability_score, 
        transcript_text=transcript_text
    )
    
    # Bruker retry-funksjonen
    response = generate_content_with_retry(model, prompt)
    
    if response and response.text:
        return response.text
    
    return ""

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

        for i, filename in enumerate(transcript_files):
            # Vis fremdrift
            print(f"[{i+1}/{len(transcript_files)}] Analyserer {os.path.basename(filename)}...")

            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 1. Hent Hovedscore
            stability_score = get_stability_score(content)
            
            # 2. Hent Drivere
            driver_raw = get_driver_analysis(content, stability_score)
            
            # Parsing av tall
            try:
                numbers = re.findall(r'-?\d+', driver_raw)
                driver_scores = [int(n) for n in numbers]
                
                if len(driver_scores) < 7:
                    driver_scores += [0] * (7 - len(driver_scores))
                else:
                    driver_scores = driver_scores[:7]     
            except Exception as e:
                print(f"    Kunne ikke parse tall fra: '{driver_raw}'")
                driver_scores = [0] * 7

            print(f"    -> Score: {stability_score}, Drivere: {driver_scores}")

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
            
            # VIKTIG: Lengre pause mellom hver fil for å unngå 429-feil igjen
            time.sleep(5) 

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