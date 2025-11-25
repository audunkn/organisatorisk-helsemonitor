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
    'Etterspørselsmønstre',
    'Prismakt',
    'Strategigjennomføring'
]

# 2. Beregne gjennomsnitt for driverne
driver_means = df[drivers].mean().sort_values()

# --- Visualisering ---

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Øk hspace for mer plass mellom plottene
plt.subplots_adjust(hspace=0.7)

## 📊 Plot 1: Fordeling av Forretningsstabilitet
# -----------------------------------------------------------------

# Undertekst for Plot 1
subtitle_text = "Stabilitet vurdert ut ifra robusthet og fremtidsutsikter"


axes[0].set_title(
    'Kvantitativ vurdering av forretningsstabilitet - antall pr kategori (-2 til +2)',
    fontsize=18,
    fontweight='bold',
    loc='center',
    y=1.05
)

# Setter undertittel/definisjon (Plot 1) - Større og ikke kursiv
axes[0].text(
    x=0.5, y=1.0, s=subtitle_text,
    ha='center', va='bottom',
    fontsize=12, style='normal', wrap=True,
    transform=axes[0].transAxes
)

# Fargepalett for Plot 1
custom_palette_plot1 = {
    -2: '#FFEC99',
    -1: '#F8A96F',
    0: '#CCCCCC',  # Nøytral grå for kategori 0
    1: '#8EC364',  # Grøntone for +1
    2: '#1A6B3D'   # Mørk grøntone for +2
}

# Definer hele rekkefølgen eksplisitt for å inkludere 0
stabilitet_order = [-2, -1, 0, 1, 2]
palette_values = [custom_palette_plot1.get(k, 'lightgrey') for k in stabilitet_order]

# Fikset: Fjernet hue='Stabilitet' for å sikre at fargene i palette_values
# matcher rekkefølgen i stabilitet_order
bar_container = sns.countplot(
    x='Stabilitet',
    data=df,
    palette=palette_values,
    order=stabilitet_order,
    ax=axes[0],
)

# Gjør x-akse benevnelsene tydeligere/større
axes[0].tick_params(axis='x', labelsize=12)

# FUNKSJON: Legger til fortegn og "Kategori: " for x-aksen (Plot 1)
def format_label(val_str):
    try:
        val = int(val_str)
        if val > 0:
            return f"Kategori: +{val}" 
        else:
            return f"Kategori: {val}"
    except:
        return f"Kategori: {val_str}"

# Henter og formaterer tick labels
current_ticks = [t.get_text() for t in axes[0].get_xticklabels()]
labels = [format_label(t) for t in current_ticks]
axes[0].set_xticklabels(labels)

for tick in axes[0].get_xticklabels():
    tick.set_fontweight('bold')

# Manuelt fjern eventuell legend
if axes[0].get_legend():
    axes[0].get_legend().remove()

# FJERNEDE LINJER: Seksjonen som la til tall over søylene er fjernet her

axes[0].set_xlabel('')
axes[0].set_ylabel('Antall møtereferater', fontsize=12)

## 📈 Plot 2: Driver-analyse
# -------------------------------------------------

# Fiks for FutureWarning: Midlertidig DataFrame for hue-basert fargelegging
temp_df = pd.DataFrame({
    'Score': driver_means.values,
    'Driver': driver_means.index
})
temp_df['Color_Hue'] = np.where(temp_df['Score'] < 0, 'Negative', 'Positive')

# Fargepalett for Plot 2
custom_palette_plot2 = {'Negative': '#E87777', 'Positive': 'lightgrey'}

sns.barplot(
    x='Score',
    y='Driver',
    data=temp_df,
    hue='Color_Hue',
    palette=custom_palette_plot2,
    ax=axes[1],
)
# Manuelt fjern eventuell legend
if axes[1].get_legend():
    axes[1].get_legend().remove()

# Fjern x-akse benevnelsen
axes[1].set_xlabel('', fontsize=12)
axes[1].set_xlim(-2, 2)
axes[1].set_ylabel('')

# Legger til verdien ved siden av baren
for i, v in enumerate(driver_means.values):
    text_x = v + 0.05 if v >= 0 else v - 0.05
    ha = 'left' if v >= 0 else 'right'
    axes[1].text(text_x, i, f'{v:.2f}', color='black', va='center', ha=ha, fontweight='bold')

# Legger til en vertikal linje ved 0
axes[1].axvline(0, color='darkgrey', linestyle='--', linewidth=1)

# Undertekst for Plot 2: "Snitt score pr driver (Alle møtereferater)" uten fet skrift
caption_text = 'Snitt score pr driver (Alle møtereferater)'
fig.text(
    x=0.5, y=0.03, s=caption_text,
    ha='center', va='bottom',
    fontsize=12, fontweight='normal',
)

# Lagre eller vise
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()