# Laboratório do capítulo 1
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. Carregar os dados ---------------------------------------------------
icu     <- read.csv("data/icu_aki.csv")
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
dim(icu)
str(icu)

# 2. Descrever --------------------------------------------------------------
summary(icu$delta_creatinine_48h)
icu$aki <- icu$delta_creatinine_48h >= 0.3          # critério KDIGO
mean(icu$aki)                                        # risco de referência 'sem modelo'
round(tapply(icu$aki, icu$site, mean), 2)            # difere entre os hospitais?

# 3. Divisão treino / teste: a divisão fixa do livro (1.000 / 250) ------------
split <- read.csv("data/splits/ward7_split.csv")
stopifnot(identical(split$patient_id, icu$patient_id))
train <- icu[split$set == "train", ]
test  <- icu[split$set == "test", ]

# 4. Dois modelos do desfecho ----------------------------------------------
# (a) o modelo da média: a mesma predição para todos
pred_avg  <- mean(train$delta_creatinine_48h)
# (b) uma reta em função do SOFA (o Capítulo 4 explica o lm em detalhe)
line_sofa <- lm(delta_creatinine_48h ~ sofa, data = train)
pred_line <- predict(line_sofa, newdata = test)

mae <- function(y, yhat) mean(abs(y - yhat))
c(average = mae(test$delta_creatinine_48h, pred_avg),
  sofa_line = mae(test$delta_creatinine_48h, pred_line))

# 5. Implantar: predições para os pacientes desta noite -----------------------
data.frame(name = tonight$name,
           sofa = tonight$sofa,
           predicted_change = round(predict(line_sofa, newdata = tonight), 2))

# 6. Gráfico ------------------------------------------------------------------
plot(jitter(icu$sofa), icu$delta_creatinine_48h, pch = 16, col = "#1f6f8b40",
     xlab = "Escore SOFA", ylab = "Variação da creatinina em 48 h (mg/dL)")
abline(line_sofa, lwd = 3)
abline(h = 0.3, lty = 2)
