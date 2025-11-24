import os
import google.generativeai as genai
from dotenv import load_dotenv

# --- KONFIGURASJON ---
# Bruk en standard modell som er garantert å fungere
MODEL_NAME = 'gemini-2.5-flash' 
TEST_PROMPT = "Hva er den første tingen du tenker på når du hører ordet 'Python'?"
# --------------------

def kjør_tilkoblingssjekk():
    """
    Sjekker tilkobling til Gemini API basert på innstillinger fra originalskriptet.
    """
    
    # 1. Last inn miljøvariabler og sjekk nøkkel
    load_dotenv() 
    API_KEY = os.getenv("GEMINI_API_KEY") 
    
    print("🚀 Starter tilkoblingstest...")

    if not API_KEY:
        # Henter feilmelding direkte fra ditt originale skript
        raise ValueError("❌ FEIL: GEMINI_API_KEY er ikke funnet i miljøvariablene. Sjekk .env filen.")

    try:
        # 2. Konfigurer klienten (brukt i ditt originale skript)
        genai.configure(api_key=API_KEY)
        
        # 3. Initialiser modellen og send et enkelt innhold
        model = genai.GenerativeModel(MODEL_NAME)
        
        print(f"   ✅ API-nøkkel funnet. Sender forespørsel til **{MODEL_NAME}**...")
        print(f"   💬 Prompt: '{TEST_PROMPT}'")
        
        response = model.generate_content(TEST_PROMPT)

        # 4. Sjekk responsen
        if response and response.text:
            print("\n🎉 **SUKSESS! Tilkobling og respons OK.**")
            print("--- Modellens Svar ---")
            # Skriver ut de første 200 tegnene av svaret
            print(response.text.strip()[:200] + "...")
            print("-----------------------")
        else:
            print("\n⚠️ Advarsel: Tilkobling OK, men responsen er tom (kan skyldes filtrering).")

    except Exception as e:
        # Fanger opp alle API- og nettverksfeil
        print(f"\n❌ **FEIL:** Klarte ikke å fullføre API-kallet.")
        print(f"Detaljer: {e}")

if __name__ == "__main__":
    kjør_tilkoblingssjekk()