import os
import glob
import pandas as pd
import google.generativeai as genai
import time
from dotenv import load_dotenv # Ny import for å lese .env

# --- KONFIGURASJON ---

# Last inn miljøvariabler fra .env fil (som er ignorert av Git)
load_dotenv() 


API_KEY = os.getenv("GEMINI_API_KEY") 

if not API_KEY:
    raise ValueError("GEMINI_API_KEY er ikke funnet. Vennligst sjekk .env filen din eller miljøvariablene.")

genai.configure(api_key=API_KEY)

MODEL_NAME = 'models/gemini-2.5-flash-preview-09-2025'
PROMPT_DIR = "prompts" # Ny konstant for prompt-mappen


# --- NY FUNKSJON FOR Å LESE PROMPTS ---
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
    """
    model = genai.GenerativeModel(MODEL_NAME)
    
    # Laster inn prompten fra fil
    prompt_template = load_prompt('business_stability_prompt.txt')
    
    # Fyller inn transkriptet i malen ved hjelp av format()
    prompt = prompt_template.format(transcript_text=transcript_text) 
    
    try:
        response = model.generate_content(prompt)
        # Rensing for å finne tallet
        clean_score = response.text.strip().replace('+', '') 
        return int(clean_score)
    except Exception as e:
        print(f"Feil under stabilitets-score: {e}")
        return 0

def get_driver_analysis(transcript_text, stability_score):
    """
    Kjører analysen ved å bruke prompt fra driver_analysis_prompt.txt
    """
    model = genai.GenerativeModel(MODEL_NAME)
    
    # Laster inn prompten fra fil
    prompt_template = load_prompt('driver_analysis_prompt.txt')

    # Fyller inn konteksten og transkriptet i malen ved hjelp av format()
    prompt = prompt_template.format(stability_score=stability_score, transcript_text=transcript_text)
    
    try:
        response = model.generate_content(prompt)
        # Rensing av tekst for å sikre at vi kun får tallene
        return response.text.strip()
    except Exception as e:
        print(f"Feil under driver-analyse: {e}")
        return "0,0,0,0,0,0,0" # Returnerer null-verdier ved feil

# Resten av main() funksjonen er uendret, men den kaller nå de nye funksjonene:
def main():
    # Finn alle .txt filer i mappen
    transcript_files = glob.glob("full_transcripts_output/*.txt")
    
    if not transcript_files:
        print("Fant ingen .txt filer i mappen 'full_transcripts_output'.")
        return

    results = []

    print(f"Fant {len(transcript_files)} filer. Starter analyse...\n")

    for filename in transcript_files:
        print(f"Analyserer {filename}...")
        
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # --- API KALL 1: Overall Stability Score ---
        stability_score = get_stability_score(content)

        # --- API KALL 2: Driver Analysis (with stability score context) ---
        driver_raw = get_driver_analysis(content, stability_score)
        # Konverterer strengen "-2, 0, 1..." til en liste med tall
        try:
            driver_scores = [int(x.strip()) for x in driver_raw.split(',')]
            # Sikre at vi har 7 verdier (fyller på med 0 hvis AI feilet litt)
            if len(driver_scores) != 7:
                driver_scores = driver_scores + [0]*(7-len(driver_scores))
        except:
            driver_scores = [0, 0, 0, 0, 0, 0, 0]

        # Samle data
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
        
        # En liten pause for å ikke treffe rate limits hvis du har mange filer
        time.sleep(2) 

    # --- LAGRE TIL TABELL ---
    df = pd.DataFrame(results)
    
    output_filename = "analyse_resultater.csv"
    df.to_csv(output_filename, index=False, sep=';') # Bruker semikolon for enklere Excel-åpning i Norge
    
    print(f"\nFerdig! Resultatene er lagret i '{output_filename}'.")
    print("-" * 30)
    print(df.head())

if __name__ == "__main__":
    main()