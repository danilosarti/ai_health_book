# Laboratório do capítulo 4
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 1. UTI Ala 7: carregar e dividir (a divisão compartilhada por todo o livro) -----
icu     <- read.csv("data/icu_aki.csv", stringsAsFactors = TRUE)
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
tonight$site <- factor(tonight$site, levels = levels(icu$site))
split_w7 <- read.csv("data/splits/ward7_split.csv")
set_w7 <- split_w7$set[match(icu$patient_id, split_w7$patient_id)]
train <- icu[set_w7 == "train", ]
test  <- icu[set_w7 == "test", ]

# 2. Regressão linear múltipla ---------------------------------------------------
f <- delta_creatinine_48h ~ site + age + sofa + lactate + map_min +
       baseline_creat + protocol_adherence + bun + crp + albumin
fit <- lm(f, data = train)
summary(fit)
round(confint(fit), 4)
drop1(fit, test = "F")              # cada termo testado por último: a ordem não importa

# 3. Diagnóstico ------------------------------------------------------------------
par(mfrow = c(2, 2)); plot(fit); par(mfrow = c(1, 1))
cd <- cooks.distance(fit)
c(max_cook = max(cd), flagged = sum(cd > 4 / nrow(train)))
X <- model.matrix(fit)[, -1]
vif <- sapply(colnames(X), function(j)
  1 / (1 - summary(lm(X[, j] ~ X[, -match(j, colnames(X))]))$r.squared))
round(vif, 2)

# 4. Erro no conjunto de teste comparado com a referência da 'média' -------------
mae  <- function(y, yhat) mean(abs(y - yhat))
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))
y_te <- test$delta_creatinine_48h
pred <- predict(fit, newdata = test)
rbind(average = c(MAE = mae(y_te, mean(train$delta_creatinine_48h)),
                  RMSE = rmse(y_te, mean(train$delta_creatinine_48h))),
      linear  = c(MAE = mae(y_te, pred), RMSE = rmse(y_te, pred)))

# 5. Os pacientes desta noite, com intervalos de predição de 95% ------------------
cbind(tonight["name"], round(predict(fit, newdata = tonight, interval = "prediction"), 2))

# 6. ONCO-1: OLS, ridge, elastic net e lasso ---------------------------------------
if (!requireNamespace("glmnet", quietly = TRUE)) install.packages("glmnet")
library(glmnet)
onco <- read.csv("data/onco1.csv")
drop <- c("patient_id", "high_response", "baseline_tumor_mm", "post_tumor_mm",
          "gene_08", "gene_14", "gene_19")      # identificador, cópias do desfecho, genes com vazamento
d <- onco[, setdiff(names(onco), drop)]
d$treatment <- factor(d$treatment, levels = c("no_chemo", "chemo"))
Xo <- model.matrix(response_percent ~ ., data = d)[, -1]
yo <- d$response_percent
split_o <- read.csv("data/splits/onco1_split.csv")
split_o <- split_o[match(onco$patient_id, split_o$patient_id), ]
tr <- which(split_o$set == "train")
foldid <- split_o$fold[tr]                         # partições compartilhadas, as mesmas para todos os modelos
Z <- scale(Xo, colMeans(Xo[tr, ]), apply(Xo[tr, ], 2, sd))   # padronizada com as médias e os DPs de treino
cvs <- list(ridge = cv.glmnet(Z[tr, ], yo[tr], alpha = 0,   foldid = foldid, standardize = FALSE,
                              lambda.min.ratio = 1e-6),  # grade longa: a trajetória da ridge chega ao OLS
            enet  = cv.glmnet(Z[tr, ], yo[tr], alpha = 0.8, foldid = foldid, standardize = FALSE),
            lasso = cv.glmnet(Z[tr, ], yo[tr], alpha = 1,   foldid = foldid, standardize = FALSE))
ols <- lm(yo[tr] ~ Z[tr, ])
test_err <- function(p) c(MAE = mae(yo[-tr], p), RMSE = rmse(yo[-tr], p))
round(cbind(OLS = test_err(cbind(1, Z[-tr, ]) %*% coef(ols)),
            sapply(cvs, function(cv) test_err(predict(cv, Z[-tr, ], s = "lambda.min"))),
            sapply(cvs, function(cv) test_err(predict(cv, Z[-tr, ], s = "lambda.1se")))), 3)
# colunas: OLS, depois ridge / elastic net (alpha = 0.8, escolhido por VC) / lasso em lambda.min e lambda.1se
plot(cvs$lasso)                                   # curva de validação cruzada com lambda.min e lambda.1se
plot(cvs$lasso$glmnet.fit, xvar = "lambda")       # trajetórias dos coeficientes
b <- as.matrix(coef(cvs$lasso, s = "lambda.1se"))
round(b[b[, 1] != 0, , drop = FALSE], 3)          # preditores mantidos pelo lasso (por DP)
