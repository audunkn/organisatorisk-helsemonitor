import streamlit as st
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score

# --- KONFIGURASJON ---
RESULTAT_FIL = 'analyse_resultater.csv'
LOGG_FIL = 'evaluering_logg.csv'
TEKST_MAPPE = 'full_transcripts_output'

KATEGORIER = [
    "Stabilitet", "Makroforhold", "Forsyningskjede", 
    "Produksjonskvalitet", "Kompetanse", "Etterspørselsmønstre", 
    "Prismakt", "Strategigjennomføring"
]

# --- HJELPEFUNKSJONER ---
def last_data():
    if not os.path.exists(RESULTAT_FIL):
        st.error(f"Mangler {RESULTAT_FIL}")
        return pd.DataFrame()
    return pd.read_csv(RESULTAT_FIL, sep=';')

def last_logg():
    if os.path.exists(LOGG_FIL):
        df = pd.read_csv(LOGG_FIL)
        if 'Kommentar' not in df.columns: df['Kommentar'] = ""
        return df
    return pd.DataFrame(columns=['Filnavn', 'Kategori', 'Model_Score', 'Human_Score', 'Kommentar'])

def beregn_metrikker(logg_df):
    if logg_df.empty: return 0, 0, 0, 0

    # --- Implementert endring: Filtrer til kun den siste (reviderte) vurderingen ---
    # Sorterer dataen implisitt etter rekkefølgen de ble lagt til (som er den siste) 
    # og beholder kun den siste vurderingen for hver unike kombinasjon av Filnavn og Kategori.
    logg_df_siste = logg_df.drop_duplicates(subset=['Filnavn', 'Kategori'], keep='last')
    
    if logg_df_siste.empty: return 0, 0, 0, 0
    
    y_true = logg_df_siste['Human_Score'].astype(int)
    y_pred = logg_df_siste['Model_Score'].astype(int)
    
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    antall_filer = logg_df_siste['Filnavn'].nunique() # Bruker den filtrerte DFen
    
    return precision, recall, accuracy, antall_filer
    # --- Slutt på implementert endring ---

def les_tekstfil(filnavn_fra_csv):
    if os.path.exists(filnavn_fra_csv): sti = filnavn_fra_csv
    else: sti = os.path.join(TEKST_MAPPE, os.path.basename(filnavn_fra_csv))
    
    if os.path.exists(sti):
        with open(sti, 'r', encoding='utf-8') as f: return f.read()
    return "⚠️ Fant ikke filen."

def nullstill_historikk():
    if os.path.exists(LOGG_FIL): os.remove(LOGG_FIL)

def format_tall(val):
    if val > 0: return f"+{val}"
    return f"{val}"

# --- APP LAYOUT ---
st.set_page_config(layout="wide", page_title="Risiko Evaluering (Blindtest)")


st.markdown("""
    <style>
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 3px solid #4a4a4a !important;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        background-color: #fcfcfc;
        padding: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("Din vurdering og begrunnelse brukes for å forbedre instruksjonene som gis til KI-løsningen.")

df = last_data()
logg_df = last_logg()

if df.empty: st.stop()

# --- SIDEPANEL ---
st.sidebar.header("Innstillinger")
skjul_ferdige = st.sidebar.checkbox("Skjul ferdig evaluerte filer", value=False)
st.sidebar.divider()
st.sidebar.header("Forankring av KI mot fasit fra Domeneekspert")
p, r, a, c = beregn_metrikker(logg_df)
c1, c2 = st.sidebar.columns(2)
c1.metric("Filer Evaluert", c) 
c2.metric("Nøyaktighet", f"{a:.1%}")
st.sidebar.metric("Presisjon", f"{p:.1%}")
st.sidebar.metric("Sensitivitet", f"{r:.1%}")
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Slett historikk og start på nytt"):
    nullstill_historikk()
    st.rerun()

# --- HOVEDVINDU ---
alle_filer = df['Filnavn'].unique()
ferdig_evaluert = logg_df['Filnavn'].unique() if not logg_df.empty else []

if skjul_ferdige:
    filer_som_vises = [f for f in alle_filer if f not in ferdig_evaluert]
    if not filer_som_vises and len(alle_filer) > 0:
        st.success("🎉 Gratulerer! Du har evaluert alle filene.")
        st.stop()
else:
    filer_som_vises = alle_filer

if len(filer_som_vises) == 0:
    st.info("Ingen filer å vise.")
    st.stop()

def format_func_fil(filnavn):
    base = os.path.basename(filnavn)
    if filnavn in ferdig_evaluert: return f"✅ {base}"
    return f"📄 {base}"

filer_som_vises = sorted(filer_som_vises, key=lambda x: x in ferdig_evaluert)
valgt_fil = st.sidebar.selectbox("Velg fil:", filer_som_vises, format_func=format_func_fil)

rad = df[df['Filnavn'] == valgt_fil].iloc[0]
tekst = les_tekstfil(valgt_fil)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Dokument: {os.path.basename(valgt_fil)}")
    st.text_area("Innhold", tekst, height=800)

with col2:
    st.subheader("Dine vurderinger")
    with st.form("eval_form"):
        nye_data = []
        options = [-2, -1, 0, 1, 2]
        
        # --- FUNKSJON FOR Å RENDRE EN VURDERINGSRAD ---
        def render_rad(kategori, valgt_fil, logg_df, tittel_suffix=""):
            ai_val = int(rad[kategori])
            
            st.markdown(f"#### {kategori} {tittel_suffix}")
            
            # Historikk/Default
            default_val = 0 
            kommentar_historikk = ""
            
            # Finner siste vurdering (hvis den finnes) for å forhåndsutfylle skjemaet
            eksisterende_rad = logg_df[(logg_df['Filnavn'] == valgt_fil) & (logg_df['Kategori'] == kategori)]
            if not eksisterende_rad.empty:
                siste_rad = eksisterende_rad.iloc[-1]
                default_val = int(siste_rad['Human_Score'])
                kommentar_historikk = siste_rad.get('Kommentar', "")

            # Radio Input
            human_val = st.radio(
                "Din vurdering", 
                options=options, 
                index=options.index(default_val),
                key=f"rad_{valgt_fil}_{kategori}", 
                format_func=format_tall,
                horizontal=True
            )
            
            # Kommentar Input (Endret til text_area og høyde)
            kommentar_tekst = st.text_area(
                "Begrunnelse:", 
                value=kommentar_historikk, 
                key=f"kommentar_{valgt_fil}_{kategori}",
                height=70
            )
            
            return ai_val, human_val, kommentar_tekst

        # --- SAMLER INN ALLE DATA ---
        
        # 1. OVERORDNET KONKLUSJON
        with st.container(border=True):
            st.info(" **Overordnet Konklusjon**")
            ai_score, human_score, comment = render_rad(KATEGORIER[0], valgt_fil, logg_df)
            nye_data.append({'Filnavn': valgt_fil, 'Kategori': KATEGORIER[0], 'Model_Score': ai_score, 'Human_Score': human_score, 'Kommentar': comment})

        st.write("")
        st.markdown("**Underliggende Drivere**")
        st.write("")

        # 2. DRIVERE
        for kat in KATEGORIER[1:]:
            ai_score, human_score, comment = render_rad(kat, valgt_fil, logg_df, tittel_suffix="(driver)")
            nye_data.append({'Filnavn': valgt_fil, 'Kategori': kat, 'Model_Score': ai_score, 'Human_Score': human_score, 'Kommentar': comment})
            st.divider()
        
        # --- LAGRING & VALIDERING ---
        if st.form_submit_button("💾 Lagre Evaluering", type="primary", use_container_width=True):
            ny_df = pd.DataFrame(nye_data)
            
            # SJEKK PÅKREVD KOMMENTAR: Sjekk om noen av de 8 kommentarene er tomme
            if (ny_df['Kommentar'].str.strip() == '').any():
                st.error("❌ Alle 8 begrunnelsesfeltene må fylles ut. Vennligst sjekk alle kategorier.")
            else:
                # Hvis alt er OK, lagre (legger til nederst i filen)
                hdr = not os.path.exists(LOGG_FIL)
                ny_df.to_csv(LOGG_FIL, mode='a', header=hdr, index=False)
                st.toast("Lagret! Listen oppdatert.", icon="✅")
                st.rerun()