# Laboratório do capítulo 6
# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde
# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.
# 0. Pacotes e dados -----------------------------------------------------------
library(rpart)
library(randomForest)
icu     <- read.csv("data/icu_aki.csv")
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
icu$site     <- factor(icu$site)
tonight$site <- factor(tonight$site, levels = levels(icu$site))
split <- read.csv("data/splits/ward7_split.csv")   # divisão compartilhada do Capítulo 3
train <- icu[split$set == "train", ]
test  <- icu[split$set == "test", ]
folds <- split$fold[split$set == "train"]          # cinco partições fixas de VC
mae   <- function(y, yhat) mean(abs(y - yhat))
f <- delta_creatinine_48h ~ site + age + sofa + lactate + map_min + baseline_creat +
       protocol_adherence + bun + crp + albumin

# 1. Cultivar uma árvore grande e podá-la com a regra 1-SE ------------------------
big  <- rpart(f, data = train, control = rpart.control(cp = 0.001, xval = folds))
cpt  <- big$cptable
best <- which.min(cpt[, "xerror"])
cp_1se <- cpt[cpt[, "xerror"] <= cpt[best, "xerror"] + cpt[best, "xstd"], "CP"][1]
small <- prune(big, cp = cp_1se)
print(small)                                 # as regras

# 2. Uma floresta aleatória (pequena, para rodar rápido) -------------------------
set.seed(2026)
rf <- randomForest(f, data = train, ntree = 300, mtry = 3)
print(rf)                                    # erro OOB

# 3. Comparar nos pacientes de teste ----------------------------------------------
lin <- lm(f, data = train)
round(c(average      = mae(test$delta_creatinine_48h, mean(train$delta_creatinine_48h)),
        big_tree     = mae(test$delta_creatinine_48h, predict(big, test)),
        pruned_tree  = mae(test$delta_creatinine_48h, predict(small, test)),
        forest       = mae(test$delta_creatinine_48h, predict(rf, test)),
        linear       = mae(test$delta_creatinine_48h, predict(lin, test))), 3)
# As árvores e o modelo linear reproduzem exatamente o registro de modelos (ledger) do livro. A
# floresta do ledger (500 árvores) e o XGBoost foram ajustados em Python (scikit-learn, xgboost),
# então esta floresta do R difere um pouco; o XGBoost não roda no navegador. O ledger guarda todos
# os modelos comparados no livro:
led <- read.csv("data/precomputed/ledger/ward7_reg.csv", encoding = "UTF-8")
led <- led[led$chapter <= 6 & led$role != "variant", c("model", "test_MAE")]
led$test_MAE <- round(led$test_MAE, 3)
led

# 4. Os pacientes desta noite -------------------------------------------------------
data.frame(name   = tonight$name,
           tree   = round(predict(small, tonight), 2),
           forest = round(predict(rf, tonight), 2))

# 5. ONCO-1: triagem de vazamentos antes de modelar --------------------------------
onco  <- read.csv("data/onco1.csv")
osp   <- read.csv("data/splits/onco1_split.csv")
otr   <- onco[osp$set == "train", ]
genes <- grep("^gene_", names(onco), value = TRUE)
r <- sapply(otr[genes], function(g) cor(g, otr$response_percent))
print(sort(round(r, 2), decreasing = TRUE)[1:5])

# 6. ...e ver o que os genes com vazamento fazem com uma árvore de regressão -----
ote  <- onco[osp$set == "test", ]
drop <- c("patient_id", "baseline_tumor_mm", "post_tumor_mm", "high_response")
keep <- setdiff(names(onco), drop)
t_all <- rpart(response_percent ~ ., data = otr[, keep])
t_cln <- rpart(response_percent ~ ., data = otr[, setdiff(keep, c("gene_08", "gene_14", "gene_19"))])
round(c(with_leaky_genes    = mae(ote$response_percent, predict(t_all, ote)),
        without_leaky_genes = mae(ote$response_percent, predict(t_cln, ote))), 2)
