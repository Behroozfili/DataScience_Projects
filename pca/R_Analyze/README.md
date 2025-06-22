# Analyse von Patientendaten und Behandlungen

Dieses Projekt enthält R-Skripte zur Analyse von Patientendaten und Behandlungsmethoden basierend auf einer fiktiven Patientendatenbank. Es werden verschiedene Analysen durchgeführt, darunter die Berechnung der Gesamtdosierung pro Behandlung, die Analyse von Behandlungskategorien und die Ermittlung von wöchentlichen Trends in den Behandlungen.

## 📊 Funktionen und Analysen

Die folgenden Hauptfunktionen und Analysen werden in diesem Skript durchgeführt:

1. **Datenvorbereitung und Zusammenführung**:
   - Es werden zwei Datensätze verwendet: `patients_dt` (Patienteninformationen) und `treatments_dt` (Behandlungsinformationen).
   - Diese werden anhand der `treatment_id` zusammengeführt, um eine umfassende Datenbasis zu schaffen.

2. **Balkendiagramme erstellen**:
   - Es werden mehrere Balkendiagramme erstellt, um verschiedene Metriken zu visualisieren:
     - Gesamtdosierung pro Behandlung.
     - Durchschnittliche Dosierung nach Kategorie.
     - Durchschnittliche Wirksamkeit nach Kategorie.
     - Tägliche Patientenbesuche.
     - Ärzte, die mindestens 3 verschiedene Kategorien verschrieben haben.
     - Behandlungen, die jede Woche verschrieben wurden.

3. **Weitere Berechnungen**:
   - Gesamtdosierung pro Behandlung.
   - Durchschnittsdosierung nach Behandlungskategorie.
   - Anzahl der Patientenbesuche pro Tag.
   - Wöchentliche Zusammenfassung der Behandlungen nach Kategorie.
   - Kumulative Dosierung pro Behandlung.
   - Identifikation von Ärzten, die mehr als 3 Behandlungskategorien verschrieben haben.

4. **Speichern von Plots**:
   - Alle Diagramme werden im `plots`-Verzeichnis als PNG-Dateien gespeichert.

## 📋 Voraussetzungen

Um das Skript auszuführen, stellen Sie sicher, dass die folgenden Pakete installiert sind:

- `data.table` – Für effiziente Datenmanipulation.
- `ggplot2` – Zum Erstellen von Visualisierungen (Balkendiagrammen).
  
Sie können diese Pakete mit dem folgenden Befehl installieren:

```r
install.packages(c("data.table", "ggplot2"))
