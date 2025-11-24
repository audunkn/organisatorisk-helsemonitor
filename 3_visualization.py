import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Laste inn data
try:
    df = pd.read_csv('analyse_resultater.csv', sep=';')
except FileNotFoundError:
    print("FEIL: Filen 'analyse_resultater.csv' ble ikke funnet. Sjekk filnavn og plassering.")
    exit()

drivers = [
    'Makroforhold', 
    'Forsyningskjede', 
    'Produksjonskvalitet', 
    'Kompetanse', 
    'Etterspørsel', 
    'Marginstyring', 
    'Strategigjennomføring'
]

# 2. Beregne gjennomsnitt for driverne
driver_means = df[drivers].mean().sort_values()

# --- Visualisering ---

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
# Justerer hspace for å sikre avstand mellom plottene
plt.subplots_adjust(hspace=0.5) 

# 📊 Plot 1: Fordeling av Forretningsstabilitet
# -----------------------------------------------------------------

# Definisjon av 'Forretningsstabilitet' (Forkortet versjon)
subtitle_text = "Forretningsstabilitet vurdert ut fra robusthet og fremtidsutsikter."

# 1. Sett hovedtittel
axes[0].set_title(
    'Forretningsstabilitet pr kategori (-2 til +2)',
    fontsize=14, 
    fontweight='bold',
    loc='center', 
    y=1.05 # Plasser den litt høyere
)

# 2. Sett undertittel/definisjon ved hjelp av axes[0].text() for bedre formatering
axes[0].text(
    x=0.5, y=1.0, s=subtitle_text, 
    ha='center', va='bottom', # Midtstilt og henger fra toppen
    fontsize=9., style='italic', wrap=True, 
    transform=axes[0].transAxes # Bruker relative koordinater
)


# Fiks for FutureWarning: Setter hue til 'Forretningsstabilitet'
sns.countplot(
    x='Forretningsstabilitet', 
    data=df, 
    palette='RdYlGn', 
    ax=axes[0],
    hue='Forretningsstabilitet', 
    legend=False                   
)



axes[0].set_ylabel('Antall referater', fontsize=12)

# Legger til tall over søylene
for p in axes[0].patches:
    if p.get_height() > 0:
        axes[0].annotate(f'{int(p.get_height())}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points')


# 📈 Plot 2: Driver-analyse 
# -------------------------------------------------

# Fargelegging: Drivere med NEGATIV gjennomsnittsscore blir røde, ellers grå
colors = ['firebrick' if v < 0 else 'lightgrey' for v in driver_means.values]

# Fiks for FutureWarning: Midlertidig DataFrame for hue-basert fargelegging
temp_df = pd.DataFrame({
    'Score': driver_means.values, 
    'Driver': driver_means.index
})
temp_df['Color_Hue'] = np.where(temp_df['Score'] < 0, 'Negative', 'Positive')
custom_palette_plot2 = {'Negative': 'firebrick', 'Positive': 'lightgrey'}

sns.barplot(
    x='Score', 
    y='Driver', 
    data=temp_df, 
    hue='Color_Hue',           
    palette=custom_palette_plot2,    
    ax=axes[1],
    legend=False               
)

axes[1].set_title('Gjennomsnittlig score per driver (Hele porteføljen)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Gjennomsnittlig Score', fontsize=12)
axes[1].set_xlim(-2, 2) 
axes[1].set_ylabel('') 

# Legger til verdien ved siden av baren
for i, v in enumerate(driver_means.values):
    text_x = v + 0.05 if v >= 0 else v - 0.05 
    ha = 'left' if v >= 0 else 'right'
    axes[1].text(text_x, i, f'{v:.2f}', color='black', va='center', ha=ha, fontweight='bold')
    
# Legger til en vertikal linje ved 0
axes[1].axvline(0, color='darkgrey', linestyle='--', linewidth=1)


# Lagre eller vise
# Bruker plt.tight_layout uten ekstra argumenter for å la Matplotlib beregne den beste passformen
plt.tight_layout() 
plt.show()