# PCA-Analyse des NCI60-Datensatzes mit sequenzieller Anzeige und Speicherung von Plots

**Autor:** Behrooz Filzadeh

## Beschreibung

Dieses R-Skript führt eine Hauptkomponentenanalyse (PCA) für den NCI60-Genexpressionsdatensatz aus dem `ISLR2`-Paket durch. Das Hauptziel ist die Dimensionsreduktion und die Visualisierung der Struktur in den Daten, insbesondere im Hinblick auf die verschiedenen Krebstypen.

Das Skript zeichnet sich durch folgende Merkmale aus:
*   Es führt eine Standard-PCA mit Skalierung und Zentrierung der Daten durch.
*   Es generiert aussagekräftige Visualisierungen:
    *   **Scree-Plot:** Zeigt den Anteil der erklärten Varianz pro Hauptkomponente sowie die kumulative Varianz.
    *   **Score-Plots:** Visualisieren die Position der Proben (Zelllinien) in Bezug auf die ersten Hauptkomponenten (z.B. PC1 vs. PC2, PC1 vs. PC3), eingefärbt nach Krebstyp und mit Legende.
    *   **"Aussagekräftiger" Biplot:** Stellt sowohl die Proben als auch die Variablen (Gene) mit dem größten Einfluss auf die ersten beiden PCs dar.
    *   **Vergleich von Biplot-Skalierungen:** Zeigt manuell erstellte Biplots, um den Effekt unterschiedlicher Skalierungen (Fokus auf Variablen vs. Fokus auf Proben) zu demonstrieren.
*   **Sequenzielle Anzeige:** Die Plots werden nacheinander im R-Grafikfenster angezeigt, wobei das Skript nach jedem Plot pausiert und auf eine Benutzereingabe ([Enter]) wartet.
*   **Speichern von Plots:** Alle erzeugten Plots können optional als hochauflösende PNG-Dateien in einem konfigurierbaren Verzeichnis gespeichert werden.
*   **Konfigurierbarkeit:** Wichtige Parameter wie das Speichern von Plots, das Zielverzeichnis und Plot-Dimensionen können einfach am Anfang des Skripts angepasst werden.
*   **Farbmanagement:** Nutzt `RColorBrewer` für eine klare und unterscheidbare Farbgebung der verschiedenen Krebstypen in den Plots, mit einem Fallback auf Standardfarben bei sehr vielen Typen.

## Abhängigkeiten

Die folgenden R-Pakete werden benötigt:

*   `ISLR2`: Für den NCI60-Datensatz.
*   `RColorBrewer`: Für bessere Farbpaletten in den Plots.

Falls diese Pakete noch nicht installiert sind, können Sie dies mit folgendem Befehl tun:
```R
install.packages(c("ISLR2", "RColorBrewer"))