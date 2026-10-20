# Chapter 6 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 0. Packages and data --------------------------------------------------------
library(rpart)
library(randomForest)
icu     <- read.csv("data/icu_aki.csv")
tonight <- read.csv("data/ward7_tonight.csv", encoding = "UTF-8")
icu$site     <- factor(icu$site)
tonight$site <- factor(tonight$site, levels = levels(icu$site))
split <- read.csv("data/splits/ward7_split.csv")   # shared split from Chapter 3
train <- icu[split$set == "train", ]
test  <- icu[split$set == "test", ]
folds <- split$fold[split$set == "train"]          # five fixed CV folds
mae   <- function(y, yhat) mean(abs(y - yhat))
f <- delta_creatinine_48h ~ site + age + sofa + lactate + map_min + baseline_creat +
       protocol_adherence + bun + crp + albumin

# 1. Grow a large tree and prune it with the 1-SE rule -----------------------------
big  <- rpart(f, data = train, control = rpart.control(cp = 0.001, xval = folds))
cpt  <- big$cptable
best <- which.min(cpt[, "xerror"])
cp_1se <- cpt[cpt[, "xerror"] <= cpt[best, "xerror"] + cpt[best, "xstd"], "CP"][1]
small <- prune(big, cp = cp_1se)
print(small)                                 # the rules

# 2. A random forest (small, so it runs quickly) -------------------------------
set.seed(2026)
rf <- randomForest(f, data = train, ntree = 300, mtry = 3)
print(rf)                                    # OOB error

# 3. Compare on the test patients -------------------------------------------------
lin <- lm(f, data = train)
round(c(average      = mae(test$delta_creatinine_48h, mean(train$delta_creatinine_48h)),
        big_tree     = mae(test$delta_creatinine_48h, predict(big, test)),
        pruned_tree  = mae(test$delta_creatinine_48h, predict(small, test)),
        forest       = mae(test$delta_creatinine_48h, predict(rf, test)),
        linear       = mae(test$delta_creatinine_48h, predict(lin, test))), 3)
# The trees and the linear model reproduce the book's model ledger exactly. The ledger's forest
# (500 trees) and XGBoost were fitted in Python (scikit-learn, xgboost), so this R forest differs a
# little; XGBoost does not run in the browser. The ledger holds every model compared in the book:
led <- read.csv("data/precomputed/ledger/ward7_reg.csv", encoding = "UTF-8")
led <- led[led$chapter <= 6 & led$role != "variant", c("model", "test_MAE")]
led$test_MAE <- round(led$test_MAE, 3)
led

# 4. Tonight's patients -------------------------------------------------------------
data.frame(name   = tonight$name,
           tree   = round(predict(small, tonight), 2),
           forest = round(predict(rf, tonight), 2))

# 5. ONCO-1: screen for leaks before modelling ----------------------------------
onco  <- read.csv("data/onco1.csv")
osp   <- read.csv("data/splits/onco1_split.csv")
otr   <- onco[osp$set == "train", ]
genes <- grep("^gene_", names(onco), value = TRUE)
r <- sapply(otr[genes], function(g) cor(g, otr$response_percent))
print(sort(round(r, 2), decreasing = TRUE)[1:5])

# 6. ...and see what the leaky genes do to a regression tree -------------------
ote  <- onco[osp$set == "test", ]
drop <- c("patient_id", "baseline_tumor_mm", "post_tumor_mm", "high_response")
keep <- setdiff(names(onco), drop)
t_all <- rpart(response_percent ~ ., data = otr[, keep])
t_cln <- rpart(response_percent ~ ., data = otr[, setdiff(keep, c("gene_08", "gene_14", "gene_19"))])
round(c(with_leaky_genes    = mae(ote$response_percent, predict(t_all, ote)),
        without_leaky_genes = mae(ote$response_percent, predict(t_cln, ote))), 2)
