# Chapter 1 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Load the data -------------------------------------------------------
icu     <- read.csv("data/icu_aki.csv")
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
dim(icu)
str(icu)

# 2. Describe ---------------------------------------------------------------
summary(icu$delta_creatinine_48h)
icu$aki <- icu$delta_creatinine_48h >= 0.3          # KDIGO criterion
mean(icu$aki)                                        # 'no model' baseline risk
round(tapply(icu$aki, icu$site, mean), 2)            # does it differ by site?

# 3. Train / test split: the book's fixed split (1 000 / 250) -----------------
split <- read.csv("data/splits/ward7_split.csv")
stopifnot(identical(split$patient_id, icu$patient_id))
train <- icu[split$set == "train", ]
test  <- icu[split$set == "test", ]

# 4. Two models of the outcome ---------------------------------------------
# (a) the average model: same prediction for everyone
pred_avg  <- mean(train$delta_creatinine_48h)
# (b) a straight line in SOFA (Chapter 4 explains lm in detail)
line_sofa <- lm(delta_creatinine_48h ~ sofa, data = train)
pred_line <- predict(line_sofa, newdata = test)

mae <- function(y, yhat) mean(abs(y - yhat))
c(average = mae(test$delta_creatinine_48h, pred_avg),
  sofa_line = mae(test$delta_creatinine_48h, pred_line))

# 5. Deploy: predictions for tonight's patients -------------------------------
data.frame(name = tonight$name,
           sofa = tonight$sofa,
           predicted_change = round(predict(line_sofa, newdata = tonight), 2))

# 6. Plot ---------------------------------------------------------------------
plot(jitter(icu$sofa), icu$delta_creatinine_48h, pch = 16, col = "#1f6f8b40",
     xlab = "SOFA score", ylab = "Change in creatinine over 48 h (mg/dL)")
abline(line_sofa, lwd = 3)
abline(h = 0.3, lty = 2)
