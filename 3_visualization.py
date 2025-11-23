import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Laste inn data
# Vi bruker semikolon som skilletegn slik CSV-filen ble generert
df = pd.read_csv('analyse_resultater.csv', sep=';')

# Definerer listen over driver-variablene (Norske navn fra CSV-en)
drivers = [
    'Makroforhold', 
    'Forsyningskjede', 
    'Produksjonskvalitet', 
    'Kompetanse', 
    'Etterspørselsmønstre', 
    'Prismakt', 
    'Strategigjennomføring'
]

# 2. Beregne gjennomsnitt for driverne for HELE datasettet
# Vi sorterer verdiene slik at de med lavest score (størst risiko) havner øverst eller til venstre
driver_means = df[drivers].mean().sort_values()

# --- Visualisering ---

# Setter stilen
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
plt.subplots_adjust(hspace=0.4)

# --- Plot 1: Fordeling av Forretningsstabilitet (Totaloversikten) ---
# NB: Sjekker at kolonnen finnes. I forrige steg kalte vi den "Forretningsstabilitet"
sns.countplot(x='Forretningsstabilitet', data=df, palette='RdYlGn', ax=axes[0])

axes[0].set_title('Fordeling: Forretningsstabilitet (Alle referater)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Score (-2 til 2)', fontsize=12)
axes[0].set_ylabel('Antall referater', fontsize=12)

# Legger til tall over søylene
for p in axes[0].patches:
    if p.get_height() > 0: # Sjekk for å unngå tekst på tomme søyler
        axes[0].annotate(f'{int(p.get_height())}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# --- Plot 2: Driver-analyse (Gjennomsnitt for alle) ---

# Fargelegging: De 3 svakeste driverne (lavest score) får rød farge for å fremheve risiko, resten grå
# Siden listen er sortert stigende, er de første elementene de laveste.
colors = ['firebrick' if i < 3 else 'lightgrey' for i in range(len(driver_means))]

sns.barplot(x=driver_means.values, y=driver_means.index, palette=colors, ax=axes[1])

axes[1].set_title('Gjennomsnittlig score per driver (Hele porteføljen)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Gjennomsnittlig Score', fontsize=12)
axes[1].set_xlim(-2, 2) # Setter fast akse fra -2 til 2 for enklere sammenligning

# Legger til verdien ved siden av baren for tydelighet
for i, v in enumerate(driver_means.values):
    axes[1].text(v + 0.05, i, f'{v:.2f}', color='black', va='center', fontweight='bold')

# Lagre eller vise
plt.tight_layout()
plt.show()