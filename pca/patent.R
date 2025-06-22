# Warnmeldungen ausschalten
options(warn = -1)

if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
library(data.table)
library(ggplot2)

# Funktion zum Erstellen von Balkendiagrammen
create_bar_plot <- function(data, x_var, y_var = NULL, title, x_label, y_label, fill_var = NULL, rotate_x = TRUE, use_count_stat = FALSE) {
  if (use_count_stat) {
    p <- ggplot(data, aes_string(x = x_var, fill = fill_var)) +
      geom_bar(stat = "count") +
      labs(title = title, x = x_label, y = y_label) +
      theme_minimal()
  } else {
    p <- ggplot(data, aes_string(x = x_var, y = y_var, fill = fill_var)) +
      geom_bar(stat = "identity") +
      labs(title = title, x = x_label, y = y_label) +
      theme_minimal()
  }
  if (rotate_x) {
    p <- p + theme(axis.text.x = element_text(angle = 45, hjust = 1))
  }
  return(p)
}

# Funktion zum Speichern von Plots im 'plots'-Ordner
save_plot <- function(plot_object, filename, width = 8, height = 5, dpi = 300) {
  if (!dir.exists("plots")) dir.create("plots")
  filepath <- file.path("plots", filename)
  ggsave(filepath, plot = plot_object, width = width, height = height, dpi = dpi)
}


patients_dt <- data.table(
  patient_id = 1:1000,
  doctor_id = sample(1:50, 1000, replace = TRUE),
  treatment_id = sample(1:20, 1000, replace = TRUE),
  dosage_mg = sample(50:500, 1000, replace = TRUE),
  visit_date = as.Date('2024-01-01') + sample(0:90, 1000, replace = TRUE)
)


treatments_dt <- data.table(
  treatment_id = 1:20,
  treatment_name = paste0("Drug_", LETTERS[1:20]),
  efficacy_score = round(runif(20, 0.5, 0.95), 2),
  category = sample(c("Antibiotics", "Anti-inflammatory", "Hormonal", "Antiviral"), 20, replace = TRUE)
)

# Struktur der Tabellen anzeigen
str(patients_dt)
str(treatments_dt)

# Zusammenführen der Patientendaten mit den Behandlungsdaten
merged_data <- merge(patients_dt, treatments_dt, by = "treatment_id")

# Gesamtdosierung pro Behandlung berechnen
total_dosage_by_treatment <- merged_data[, .(total_dosage = sum(dosage_mg)), by = treatment_id]


# Top 5 Behandlungen mit der höchsten Gesamtdosis
top5_treatments <- total_dosage_by_treatment[order(-total_dosage)][1:5]
print(top5_treatments)

# Durchschnittsdosierung nach Behandlungskategorie berechnen
average_dosage_by_category <- merged_data[, .(average_dosage = mean(dosage_mg)), by = category]

# Ermittlung der Kategorie mit der höchsten durchschnittlichen Wirksamkeit
highest_efficacy_category <- treatments_dt[, .(average_efficacy = mean(efficacy_score)), by = category]
top_category <- highest_efficacy_category[order(-average_efficacy)][1]
print(top_category)

# Anzahl der Patientenbesuche pro Tag berechnen
daily_patient_visits <- merged_data[, .(visit_count = .N), by = visit_date]
max_visit_day <- daily_patient_visits[order(-visit_count)][1]
print(max_visit_day)

# Kalenderwoche aus Besuchsdatum extrahieren
merged_data[, week := as.integer(format(visit_date, "%U"))]

# Wöchentliche Zusammenfassung der Behandlungen nach Kategorie
weekly_summary <- merged_data[, .(weekly_treatment_count = .N), by = .(week, category)]
print(weekly_summary)



# Kumulative Dosierung pro Behandlung berechnen
merged_data[, running_total_dosage := cumsum(dosage_mg), by = treatment_id]

# Ärzte identifizieren, die mindestens 3 verschiedene Kategorien verschrieben haben
doctor_treatment_categories <- merged_data[, .(unique_categories = unique(category)), by = doctor_id]
doctors_with_multiple_categories <- doctor_treatment_categories[length(unique_categories) >= 3]
print(doctors_with_multiple_categories)



# Behandlungen identifizieren, die jede Woche verschrieben wurden
weekly_treatment_summary <- merged_data[, .(weekly_prescription_count = .N), by = .(week, treatment_id)]
treatments_prescribed_every_week <- weekly_treatment_summary[, .(prescribed_every_week = all(weekly_prescription_count > 0)), by = treatment_id]
treatments_prescribed_every_week <- treatments_prescribed_every_week[prescribed_every_week == TRUE]
print(treatments_prescribed_every_week)

plot1 <- create_bar_plot(
  total_dosage_by_treatment,
  x_var = "treatment_id",
  y_var = "total_dosage",
  title = "Gesamtdosierung pro Behandlung",
  x_label = "Behandlungs-ID",
  y_label = "Gesamtdosis (mg)",
  fill_var = NULL,
  rotate_x = FALSE
)

plot2 <- create_bar_plot(
  average_dosage_by_category,
  x_var = "category",
  y_var = "average_dosage",
  title = "Durchschnittliche Dosierung nach Kategorie",
  x_label = "Behandlungskategorie",
  y_label = "Durchschnittsdosis (mg)",
  fill_var = "category"
)

plot3 <- create_bar_plot(
  highest_efficacy_category,
  x_var = "category",
  y_var = "average_efficacy",
  title = "Durchschnittliche Wirksamkeit nach Kategorie",
  x_label = "Behandlungskategorie",
  y_label = "Durchschnittliche Wirksamkeit",
  fill_var = "category"
)

plot4 <- create_bar_plot(
  daily_patient_visits,
  x_var = "visit_date",
  y_var = "visit_count",
  title = "Tägliche Patientenbesuche",
  x_label = "Datum",
  y_label = "Anzahl der Besuche",
  fill_var = NULL
)

plot5 <- create_bar_plot(
  doctors_with_multiple_categories,
  x_var = "doctor_id",
  title = "Ärzte mit ≥ 3 verschiedenen Behandlungskategorien",
  x_label = "Arzt-ID",
  y_label = "Anzahl der Ärzte",
  fill_var = NULL,
  use_count_stat = TRUE
)

plot6 <- create_bar_plot(
  treatments_prescribed_every_week,
  x_var = "treatment_id",
  title = "Wöchentlich verschriebene Behandlungen",
  x_label = "Behandlungs-ID",
  y_label = "Anzahl der Behandlungen",
  fill_var = NULL,
  use_count_stat = TRUE
)

save_plot(plot1, "plot1_total_dosage_per_treatment.png")
save_plot(plot2, "plot2_average_dosage_by_category.png")
save_plot(plot3, "plot3_average_efficacy_by_category.png")
save_plot(plot4, "plot4_daily_patient_visits.png", width = 10) 
save_plot(plot5, "plot5_doctors_with_multiple_categories.png")
save_plot(plot6, "plot6_treatments_prescribed_every_week.png")

print("Plots wurden erfolgreich gespeichert.")


