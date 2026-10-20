# Laboratorio del capitolo 6
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 0. Pacchetti e dati -----------------------------------------------------------
library(rpart)
library(randomForest)
icu     <- read.csv("data/icu_aki.csv")
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
icu$site     <- factor(icu$site)
tonight$site <- factor(tonight$site, levels = levels(icu$site))
split <- read.csv("data/splits/ward7_split.csv")   # suddivisione condivisa del Capitolo 3
train <- icu[split$set == "train", ]
test  <- icu[split$set == "test", ]
folds <- split$fold[split$set == "train"]          # cinque blocchi fissi di VC
mae   <- function(y, yhat) mean(abs(y - yhat))
f <- delta_creatinine_48h ~ site + age + sofa + lactate + map_min + baseline_creat +
       protocol_adherence + bun + crp + albumin

# 1. Far crescere un grande albero e potarlo con la regola 1-SE ------------------
big  <- rpart(f, data = train, control = rpart.control(cp = 0.001, xval = folds))
cpt  <- big$cptable
best <- which.min(cpt[, "xerror"])
cp_1se <- cpt[cpt[, "xerror"] <= cpt[best, "xerror"] + cpt[best, "xstd"], "CP"][1]
small <- prune(big, cp = cp_1se)
print(small)                                 # le regole

# 2. Una foresta casuale (piccola, così gira in fretta) --------------------------
set.seed(2026)
rf <- randomForest(f, data = train, ntree = 300, mtry = 3)
print(rf)                                    # errore OOB

# 3. Confronto sui pazienti di test -----------------------------------------------
lin <- lm(f, data = train)
round(c(average      = mae(test$delta_creatinine_48h, mean(train$delta_creatinine_48h)),
        big_tree     = mae(test$delta_creatinine_48h, predict(big, test)),
        pruned_tree  = mae(test$delta_creatinine_48h, predict(small, test)),
        forest       = mae(test$delta_creatinine_48h, predict(rf, test)),
        linear       = mae(test$delta_creatinine_48h, predict(lin, test))), 3)
# Gli alberi e il modello lineare riproducono esattamente il registro dei modelli (ledger) del libro.
# La foresta del ledger (500 alberi) e XGBoost sono stati addestrati in Python (scikit-learn,
# xgboost), quindi questa foresta in R è un po' diversa; XGBoost non gira nel browser. Il ledger
# contiene tutti i modelli confrontati nel libro:
led <- read.csv("data/precomputed/ledger/ward7_reg.csv", encoding = "UTF-8")
led <- led[led$chapter <= 6 & led$role != "variant", c("model", "test_MAE")]
led$test_MAE <- round(led$test_MAE, 3)
led

# 4. I pazienti di stasera ----------------------------------------------------------
data.frame(name   = tonight$name,
           tree   = round(predict(small, tonight), 2),
           forest = round(predict(rf, tonight), 2))

# 5. ONCO-1: screening delle fughe di informazioni prima di modellare -----------
onco  <- read.csv("data/onco1.csv")
osp   <- read.csv("data/splits/onco1_split.csv")
otr   <- onco[osp$set == "train", ]
genes <- grep("^gene_", names(onco), value = TRUE)
r <- sapply(otr[genes], function(g) cor(g, otr$response_percent))
print(sort(round(r, 2), decreasing = TRUE)[1:5])

# 6. ...e vedere che cosa fanno i geni "che trapelano" a un albero di regressione
ote  <- onco[osp$set == "test", ]
drop <- c("patient_id", "baseline_tumor_mm", "post_tumor_mm", "high_response")
keep <- setdiff(names(onco), drop)
t_all <- rpart(response_percent ~ ., data = otr[, keep])
t_cln <- rpart(response_percent ~ ., data = otr[, setdiff(keep, c("gene_08", "gene_14", "gene_19"))])
round(c(with_leaky_genes    = mae(ote$response_percent, predict(t_all, ote)),
        without_leaky_genes = mae(ote$response_percent, predict(t_cln, ote))), 2)
