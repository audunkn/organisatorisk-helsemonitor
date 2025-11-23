import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- KONFIGURASJON ---
# Denne linjen sikrer at API-nøkkelen fra .env-filen lastes inn
load_dotenv() 

try:
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY er ikke funnet. Sjekk .env-filen.")
    
    genai.configure(api_key=API_KEY)

    print("✅ Tilgjengelige modeller (filtrert til innholdsgenerering):")
    print("---------------------------------------------------------")
    found_flash = False
    
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            if 'gemini-1.5-flash' in m.name:
                found_flash = True
    
    if not found_flash:
        print("\n⚠️ Fant ingen Gemini 1.5 Flash-modeller. Sjekk API-tilgang/region.")

except Exception as e:
    print(f"\n❌ FEIL under lasting av modeller: {e}")
    print("Sjekk at din GEMINI_API_KEY er gyldig og at python-bibliotekene er installert.")