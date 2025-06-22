# Laden notwendiger Pakete (optional, aber gute Praxis für Reproduzierbarkeit)
# library(grDevices) # Für png(), dev.off() - ist aber standardmäßig geladen

# ========= Konfiguration =========

# Zielordner für die Plots
output_dir <- "bernuli_plot"
# Sicherstellen, dass der Ordner existiert, sonst erstellen
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  message("Ordner erstellt: ", output_dir)
} else {
  message("Ordner existiert bereits: ", output_dir)
}

# Parameter für die Bernoulli-Verteilung
bern_p <- 1/4

# Samen für Reproduzierbarkeit setzen
set.seed(123)

# Konstanten für die Normalverteilung
n_samples <- 1000
norm_mean <- 0
norm_sd <- 1

# Konstanten für die Transformation
trans_scale <- sqrt(0.01) # Entspricht sd = 0.1
trans_shift <- 45

# Konstanten für die Gleichverteilung
unif_min <- 0
unif_max <- 1


# ========= Hilfsfunktionen für Plots =========

#' Funktion zum Speichern von Plots mit spezifischen Einstellungen
#'
#' @param filename Der Dateiname (ohne Pfad) für den Plot.
#' @param plot_code Ein Ausdruck (expression), der den Plot-Code enthält.
#' @param dir Das Verzeichnis, in dem gespeichert werden soll.
#' @param width Die Breite des Plots in Pixeln.
#' @param height Die Höhe des Plots in Pixeln.
#' @param ... Zusätzliche Argumente für par() vor dem Plotten.
save_plot <- function(filename, plot_code, dir = output_dir, width = 800, height = 600, ...) {
  filepath <- file.path(dir, filename)
  png(filename = filepath, width = width, height = height, units = "px", res = 100) # PNG-Gerät öffnen
  
  # Setze grafische Parameter (falls vorhanden)
  old_par <- par(no.readonly = TRUE) # Aktuelle Einstellungen sichern
  on.exit(par(old_par))              # Beim Verlassen der Funktion wiederherstellen
  par(...)                           # Neue Parameter setzen (z.B. mfrow)
  
  # Plot-Code ausführen
  eval(plot_code)
  
  dev.off() # Grafikgerät schließen
  message("Plot gespeichert: ", filepath)
}


# ========= Bernoulli-Verteilung Plots (PMF & CDF nebeneinander) =========

plot_code_bernoulli <- quote({
  # --- Plot 1: Wahrscheinlichkeitsmassfunktion (PMF) ---
  x_bern <- c(0, 1)  # Mögliche Ergebnisse
  pmf_bern <- dbinom(x_bern, size = 1, prob = bern_p)  # Wahrscheinlichkeiten
  
  plot(x_bern, pmf_bern,
       type = "h",            # Vertikale Linien
       lwd = 4,               # Linienbreite
       col = "steelblue",     # Linienfarbe
       ylim = c(0, 1),        # y-Achse von 0 bis 1
       xlab = "Ergebnis (k)",
       ylab = "Wahrscheinlichkeit P(X=k)",
       main = paste("Bernoulli PMF (p =", format(bern_p, digits = 2), ")"),
       xaxt = "n",            # Standard-x-Achsenbeschriftung aus
       yaxt = "n",            # Standard-y-Achsenbeschriftung aus
       cex.main = 1.0, cex.lab = 0.9, cex.axis = 0.9 # Schriftgrößen anpassen
  )
  axis(1, at = x_bern, labels = c("0", "1"))       # x-Achse bei 0 und 1
  axis(2, at = seq(0, 1, 0.25), las = 1)          # y-Achse (las=1 für horizontale Labels)
  text(x_bern, pmf_bern + 0.05, labels = round(pmf_bern, 2), col = "black", cex = 0.8)
  
  # --- Plot 2: Kumulative Verteilungsfunktion (CDF) ---
  x_vals_cdf <- seq(-0.5, 1.5, by = 0.01) # Feinere Auflösung für CDF
  cdf_vals_bern <- pbinom(x_vals_cdf, size = 1, prob = bern_p) # CDF berechnen
  
  plot(x_vals_cdf, cdf_vals_bern,
       type = "s",              # Treppenlinien
       lwd = 2,
       col = "darkgreen",
       ylim = c(-0.05, 1.05),   # Etwas Rand oben/unten
       xlab = "Ergebnis (x)",
       ylab = "Kumulative Wahrscheinlichkeit P(X <= x)",
       main = paste("Bernoulli CDF (p =", format(bern_p, digits = 2), ")"),
       xaxt = "n",
       yaxt = "n",
       cex.main = 1.0, cex.lab = 0.9, cex.axis = 0.9
  )
  axis(1, at = c(0, 1), labels = c("0", "1"))        # x-Achse
  axis(2, at = seq(0, 1, 0.25), las = 1)             # y-Achse
  # Sprungstellen markieren
  points(c(0, 1), pbinom(c(0, 1), size = 1, prob = bern_p), pch = 16, col = "red", cex = 1.2)  # gefüllte Punkte (Ende des Sprungs)
  points(c(0, 1), c(0, 1 - bern_p), pch = 1, col = "red", cex = 1.2) # offene Kreise (Anfang des Sprungs)
})

# Plot speichern
save_plot(filename = "01_bernoulli_pmf_cdf.png",
          plot_code = plot_code_bernoulli,
          width = 1200, height = 550, # Breite angepasst für zwei Plots
          mfrow = c(1, 2),             # 1 Zeile, 2 Spalten
          mar = c(4, 4, 3, 1) + 0.1)   # Ränder anpassen (unten, links, oben, rechts)


# ========= Normalverteilung N(0,1) Plots (Histogramm & Dichte) =========

# Daten generieren
x_norm <- rnorm(n_samples, mean = norm_mean, sd = norm_sd)

plot_code_norm_x <- quote({
  # --- Histogramm von x ---
  hist(x_norm,
       breaks = 30,
       col = "lightblue",
       main = "Histogramm von x ~ N(0,1)",
       xlab = "x",
       ylab = "Dichte",  # Angepasst, da probability=TRUE
       probability = TRUE, # Zeigt Dichte statt Häufigkeit
       cex.main = 1.0, cex.lab = 0.9, cex.axis = 0.9)
  # Überlagerte theoretische Dichte (optional, aber oft nützlich)
  curve(dnorm(x, mean = norm_mean, sd = norm_sd), add = TRUE, col = "red", lwd = 2)
  
  # --- Dichteplot von x ---
  plot(density(x_norm),
       col = "darkblue",
       lwd = 2,
       main = "Dichteplot von x ~ N(0,1)",
       xlab = "x",
       ylab = "Geschätzte Dichte",
       cex.main = 1.0, cex.lab = 0.9, cex.axis = 0.9)
  # Überlagerte theoretische Dichte (optional)
  curve(dnorm(x, mean = norm_mean, sd = norm_sd), add = TRUE, col = "red", lty = 2) # gestrichelt
})

# Plot speichern
save_plot(filename = "02_normal_N01_hist_density.png",
          plot_code = plot_code_norm_x,
          width = 1200, height = 550,
          mfrow = c(1, 2),
          mar = c(4, 4, 3, 1) + 0.1)


# ========= Transformierte Normalverteilung Plots (Histogramm & Dichte) =========

# Transformation anwenden
y_transformed <- trans_scale * x_norm + trans_shift
# Erwarteter Mittelwert und SD von y
y_mean_expected <- trans_scale * norm_mean + trans_shift
y_sd_expected <- trans_scale * norm_sd

plot_code_norm_y <- quote({
  # --- Histogramm von y ---
  hist(y_transformed,
       breaks = 30,
       col = "lightgreen",
       main = paste("Histogramm von y (transformiert, N(", round(y_mean_expected,1), ",", round(y_sd_expected,2), "^2))", sep=""),
       xlab = "y",
       ylab = "Dichte",
       probability = TRUE,
       cex.main = 1.0, cex.lab = 0.9, cex.axis = 0.9)
  # Überlagerte theoretische Dichte
  curve(dnorm(x, mean = y_mean_expected, sd = y_sd_expected), add = TRUE, col = "darkred", lwd = 2)
  
  
  # --- Dichteplot von y ---
  plot(density(y_transformed),
       col = "darkgreen",
       lwd = 2,
       main = paste("Dichteplot von y (transformiert, N(", round(y_mean_expected,1), ",", round(y_sd_expected,2), "^2))", sep=""),
       xlab = "y",
       ylab = "Geschätzte Dichte",
       cex.main = 1.0, cex.lab = 0.9, cex.axis = 0.9)
  # Überlagerte theoretische Dichte
  curve(dnorm(x, mean = y_mean_expected, sd = y_sd_expected), add = TRUE, col = "darkred", lty = 2)
})

# Plot speichern
save_plot(filename = "03_normal_transformed_hist_density.png",
          plot_code = plot_code_norm_y,
          width = 1200, height = 550,
          mfrow = c(1, 2),
          mar = c(4, 4, 3, 1) + 0.1)


# ========= Gleichverteilung U(0, 1) Plots (Dichte & CDF) =========

# --- Plot Dichtefunktion ---
plot_code_unif_pdf <- quote({
  x_unif_density <- seq(unif_min - 0.5, unif_max + 0.5, by = 0.001) # Etwas Rand
  y_unif_density <- dunif(x_unif_density, min = unif_min, max = unif_max)
  
  plot(x_unif_density, y_unif_density,
       type = "l",            # Linienplot
       lwd = 3,               # Etwas dicker für Klarheit
       col = "blue",
       ylim = c(0, max(y_unif_density) * 1.2), # Y-Achse etwas höher
       main = paste("Dichtefunktion der Gleichverteilung U(", unif_min, ",", unif_max, ")", sep=""),
       xlab = "x",
       ylab = "Dichte f(x)",
       cex.main = 1.0, cex.lab = 0.9, cex.axis = 0.9)
  # Linien an den Rändern hinzufügen für Klarheit
  segments(x0 = c(unif_min - 0.5, unif_max), x1 = c(unif_min, unif_max + 0.5),
           y0 = c(0, 0), y1 = c(0, 0), col = "blue", lwd = 3)
  segments(x0 = unif_min, x1 = unif_max, y0 = 1 / (unif_max - unif_min), y1 = 1 / (unif_max - unif_min), col = "blue", lwd = 3)
})

# Plot speichern
save_plot(filename = "04_uniform_pdf.png",
          plot_code = plot_code_unif_pdf,
          width = 700, height = 550, # Standardgröße für Einzelplot
          mar = c(4, 4, 3, 1) + 0.1)


# --- Plot Verteilungsfunktion (CDF) ---
plot_code_unif_cdf <- quote({
  x_unif_cdf <- seq(unif_min - 0.5, unif_max + 0.5, by = 0.01) # Bereich für CDF
  y_unif_cdf <- punif(x_unif_cdf, min = unif_min, max = unif_max) # CDF-Werte
  
  plot(x_unif_cdf, y_unif_cdf,
       type = "l",
       lwd = 3,
       col = "darkgreen",
       ylim = c(-0.05, 1.05), # Y-Achse mit Rand
       main = paste("Verteilungsfunktion der U(", unif_min, ",", unif_max, ")", sep=""),
       xlab = "x",
       ylab = "Kumulative Wahrscheinlichkeit F(x) = P(X <= x)",
       cex.main = 1.0, cex.lab = 0.9, cex.axis = 0.9)
  # Achsen für Klarheit hervorheben
  abline(h = c(0, 1), col = "grey", lty = 2)
  abline(v = c(unif_min, unif_max), col = "grey", lty = 2)
})

# Plot speichern
save_plot(filename = "05_uniform_cdf.png",
          plot_code = plot_code_unif_cdf,
          width = 700, height = 550,
          mar = c(4, 4, 3, 1) + 0.1)

message("\nAlle Plots wurden im Ordner '", output_dir, "' gespeichert.")