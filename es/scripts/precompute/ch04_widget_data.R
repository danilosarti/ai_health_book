# Chapter 4: small data for the "drag a patient" widget.
# 30 Ward 7 patients (fixed seed) with SOFA and the 48-hour change in creatinine.
# Run from the book folder:  Rscript scripts/precompute/ch04_widget_data.R
icu <- read.csv("data/icu_aki.csv")
set.seed(404)
rows <- sort(sample(nrow(icu), 30))
pts <- data.frame(patient_id = icu$patient_id[rows], sofa = icu$sofa[rows],
                  delta_creatinine_48h = icu$delta_creatinine_48h[rows])
dir.create("data/precomputed/ch04", recursive = TRUE, showWarnings = FALSE)
write.csv(pts, "data/precomputed/ch04/drag_points.csv", row.names = FALSE)
print(coef(lm(delta_creatinine_48h ~ sofa, data = pts))); print(range(pts$sofa))
