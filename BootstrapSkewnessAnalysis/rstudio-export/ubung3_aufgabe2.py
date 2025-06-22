
"""
a) Daten laden und untersuchen
Zuerst lade ich die Cacao-Daten aus dem Ordner "Datasets". 
In diesem Beispiel gehe ich davon aus, dass die Datei "cacao_data.csv" heißt. 
Um die Struktur der Daten zu prüfen, benutze ich die pandas-Bibliothek.

"""
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# Ich lade die Daten aus der CSV-Datei
data = pd.read_csv("cacao.csv")

# Ich schaue mir die ersten 5 Zeilen der Daten an
print(data.head())  # Gibt die ersten 5 Zeilen der Tabelle aus

# Ich prüfe die Struktur der Daten (Welche Spalten gibt es? Welche Datentypen haben die Spalten?)
print(data.info())  # Zeigt mir die Struktur der Daten an

"""
Erklärung über Dataset
Datenbeschreibung: Der Datensatz enthält Informationen über Kakaopflanzen. 
Die Hauptfrage der Forschung ist, 
wie verschiedene Umweltfaktoren die Kakaoproduktion beeinflussen.
Beobachtungsobjekt: Die Kakaopflanzen sind das Hauptbeobachtungsobjekt des Datensatzes.
Die Daten enthalten verschiedene Messungen, die an den Pflanzen vorgenommen wurden.

"""
"""
b) Was ist Kurtosis (Wölbung) und wie wird sie berechnet?
Kurtosis ist ein Maß dafür, wie stark die Verteilung einer Zufallsvariablen in den "Schwänzen" von der Normalverteilung abweicht.
Sie misst also die "Schwänze" einer Verteilung, also wie viele Ausreißer oder extreme Werte es gibt.
Positive Kurtosis: Die Verteilung hat „schwerere Schwänze“, was bedeutet, dass es mehr Ausreißer oder extreme Werte gibt.
Negative Kurtosis: Die Verteilung hat „leichtere Schwänze“, was darauf hinweist, dass es weniger extreme Werte gibt.
Kurtosis von null: Die Verteilung hat die gleiche Form wie die Normalverteilung, ohne außergewöhnliche Ausreißer.

Ich benutze die Funktion stats.kurtosis() aus dem scipy-Paket, um die Kurtosis zu berechnen.
"""


# Ich schaue mir die Hilfestellung zur Funktion kurtosis an
help(stats.kurtosis)

# Ich berechne die Kurtosis für die 'canopy'-Spalte (oder eine andere relevante Spalte)
kurt = stats.kurtosis(data['canopy'])
print(f"Kurtosis: {kurt}")

"""
c) Bootstrap-Verfahren zur Schätzung der Kurtosis
Jetzt werde ich die Daten filtern, bei denen ant_exclusion == 1 ist, und dann die Kurtosis berechnen. Danach führe ich eine Bootstrap-Analyse durch, um die Schätzung der Kurtosis zu verbessern.

Filtern der Daten für den Fall ant_exclusion == 1.

Berechnen der Kurtosis.

Bootstrapping zur Schätzung der Kurtosis durch Resampling der Daten.
"""


# Ich filtere die Daten, bei denen 'ant_exclusion == 1'
filtered_data = data[data['ant_exclusion'] == 1]

# Ich wähle die 'canopy'-Spalte aus (diese kann durch die entsprechende Spalte ersetzt werden)
canopy_data = filtered_data['canopy']

# Ich berechne die Kurtosis der 'canopy'-Daten
original_kurtosis = stats.kurtosis(canopy_data)
print(f"Original Kurtosis: {original_kurtosis}")

# Bootstrap-Verfahren
B = 1000  # Anzahl der Bootstrap-Stichproben
bootstrap_kurtosis = []

# Ich führe das Bootstrapping durch
np.random.seed(123)  # Ich setze den Zufallszahlengenerator, damit ich immer die gleichen Ergebnisse bekomme
for _ in range(B):
    sample = np.random.choice(canopy_data, size=len(canopy_data), replace=True)
    bootstrap_kurtosis.append(stats.kurtosis(sample))

# Ich zeichne ein Histogramm der Bootstrap-Kurtosen
plt.hist(bootstrap_kurtosis, bins=30, color='skyblue', edgecolor='black')
plt.title("Bootstrap-Verteilung der Kurtosis")
plt.xlabel("Kurtosis")
plt.ylabel("Häufigkeit")
plt.show()

# Ich berechne das 95%-Konfidenzintervall für die Kurtosis
ci_lower, ci_upper = np.percentile(bootstrap_kurtosis, [2.5, 97.5])
print(f"95%-Bootstrap-Konfidenzintervall: ({ci_lower}, {ci_upper})")
"""
Interpretation:
Kurtosis: Die berechnete Kurtosis zeigt uns, wie "schwanzlastig" die Verteilung ist. 
Eine positive Kurtosis bedeutet, dass es mehr Ausreißer als in einer Normalverteilung gibt. 
Eine negative Kurtosis zeigt an, dass die Verteilung weniger Ausreißer hat.
Konfidenzintervall: Das 95%-Konfidenzintervall gibt den Bereich an, in dem wir mit 95%iger Sicherheit davon ausgehen können, dass der wahre Wert der Kurtosis liegt.
"""
