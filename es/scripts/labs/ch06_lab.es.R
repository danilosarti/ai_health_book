# Laboratorio del capítulo 6
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 0. Paquetes y datos ------------------------------------------------------------
library(rpart)
library(randomForest)
icu     <- read.csv("data/icu_aki.csv")
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
icu$site     <- factor(icu$site)
tonight$site <- factor(tonight$site, levels = levels(icu$site))
split <- read.csv("data/splits/ward7_split.csv")   # división compartida del Capítulo 3
train <- icu[split$set == "train", ]
test  <- icu[split$set == "test", ]
folds <- split$fold[split$set == "train"]          # cinco particiones fijas de VC
mae   <- function(y, yhat) mean(abs(y - yhat))
f <- delta_creatinine_48h ~ site + age + sofa + lactate + map_min + baseline_creat +
       protocol_adherence + bun + crp + albumin

# 1. Hacer crecer un árbol grande y podarlo con la regla 1-SE ----------------------
big  <- rpart(f, data = train, control = rpart.control(cp = 0.001, xval = folds))
cpt  <- big$cptable
best <- which.min(cpt[, "xerror"])
cp_1se <- cpt[cpt[, "xerror"] <= cpt[best, "xerror"] + cpt[best, "xstd"], "CP"][1]
small <- prune(big, cp = cp_1se)
print(small)                                 # las reglas

# 2. Un bosque aleatorio (pequeño, para que se ejecute deprisa) -------------------
set.seed(2026)
rf <- randomForest(f, data = train, ntree = 300, mtry = 3)
print(rf)                                    # error OOB

# 3. Comparar en los pacientes de prueba ------------------------------------------
lin <- lm(f, data = train)
round(c(average      = mae(test$delta_creatinine_48h, mean(train$delta_creatinine_48h)),
        big_tree     = mae(test$delta_creatinine_48h, predict(big, test)),
        pruned_tree  = mae(test$delta_creatinine_48h, predict(small, test)),
        forest       = mae(test$delta_creatinine_48h, predict(rf, test)),
        linear       = mae(test$delta_creatinine_48h, predict(lin, test))), 3)
# Los árboles y el modelo lineal reproducen exactamente el registro de modelos (ledger) del libro.
# El bosque del ledger (500 árboles) y XGBoost se ajustaron en Python (scikit-learn, xgboost), así
# que este bosque de R difiere un poco; XGBoost no se ejecuta en el navegador. El ledger guarda
# todos los modelos comparados en el libro:
led <- read.csv("data/precomputed/ledger/ward7_reg.csv", encoding = "UTF-8")
led <- led[led$chapter <= 6 & led$role != "variant", c("model", "test_MAE")]
led$test_MAE <- round(led$test_MAE, 3)
led

# 4. Los pacientes de esta noche ---------------------------------------------------
data.frame(name   = tonight$name,
           tree   = round(predict(small, tonight), 2),
           forest = round(predict(rf, tonight), 2))

# 5. ONCO-1: cribado de fugas antes de modelar ------------------------------------
onco  <- read.csv("data/onco1.csv")
osp   <- read.csv("data/splits/onco1_split.csv")
otr   <- onco[osp$set == "train", ]
genes <- grep("^gene_", names(onco), value = TRUE)
r <- sapply(otr[genes], function(g) cor(g, otr$response_percent))
print(sort(round(r, 2), decreasing = TRUE)[1:5])

# 6. ...y ver qué hacen los genes con fuga en un árbol de regresión ---------------
ote  <- onco[osp$set == "test", ]
drop <- c("patient_id", "baseline_tumor_mm", "post_tumor_mm", "high_response")
keep <- setdiff(names(onco), drop)
t_all <- rpart(response_percent ~ ., data = otr[, keep])
t_cln <- rpart(response_percent ~ ., data = otr[, setdiff(keep, c("gene_08", "gene_14", "gene_19"))])
round(c(with_leaky_genes    = mae(ote$response_percent, predict(t_all, ote)),
        without_leaky_genes = mae(ote$response_percent, predict(t_cln, ote))), 2)
