import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Laste inn data
# Vi antar at filen ligger i samme mappe og bruker semikolon som skilletegn
df = pd.read_csv('analyse_resultater.csv', sep=';')

# Definerer listen over driver-variablene (utelater filnavn og totalscore)
drivers = ['Macro_Env', 'Supply_Chain', 'Manufacturing_Quality', 'Human_Capital', 
           'Demand_Patterns', 'Pricing_Power', 'Strategic_Execution']

# 2. Definere utvalget: "Low Stability"
# Vi filtrerer ut alle referater som har en Overall Score lavere enn 2 (maks).
# Dette fanger opp alt fra små problemer (1) til kriser (-1/0).
low_stability_df = df[df['Overall_Stability_Score'] < 2]

# 3. Beregne gjennomsnitt for driverne i denne gruppen
# Dette gir oss "stemningen" for hver kategori kun i de problematiske møtene.
driver_means = low_stability_df[drivers].mean().sort_values()

# --- Visualisering ---

# Setter stilen
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
plt.subplots_adjust(hspace=0.4)

# Plot 1: Fordeling av Overall Score (Totaloversikten)
sns.countplot(x='Overall_Stability_Score', data=df, palette='RdYlGn', ax=axes[0])
axes[0].set_title('Oversikt: Overall Business Stability Score', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Score (-2 til 2)', fontsize=12)
axes[0].set_ylabel('Antall referater', fontsize=12)

# Legger til tall over søylene
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Plot 2: Driverne for lav stabilitet
# Fargelegging: De 3 nederste (verste) får rød farge, resten grå
colors = ['red' if i < 3 else 'grey' for i in range(len(driver_means))]

sns.barplot(x=driver_means.values, y=driver_means.index, palette=colors, ax=axes[1])
axes[1].set_title('Hovedårsaker til lav stabilitet (Score < 2)\n(Gjennomsnittsscore for drivere)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Gjennomsnittlig Score', fontsize=12)

# Lagre eller vise
plt.tight_layout()
plt.show()