# Laboratorio del capítulo 7
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 1. Cargar las imágenes y las predicciones precalculadas de la CNN ----------
sub  <- read.csv("data/pneumonia_subset.csv")
pred <- read.csv("data/precomputed/ch07/test_predictions.csv")
px    <- grep("^px", names(sub))                # las 784 columnas de píxeles
train <- sub[sub$split == "train", ]
test  <- sub[sub$split == "test", ]
table(split = sub$split, label = sub$label)    # 0 = normal, 1 = neumonía

# 2. Una imagen es una matriz: un tensor de orden 2 ---------------------------
to_image <- function(row) matrix(as.numeric(row), nrow = 28, ncol = 28, byrow = TRUE) / 255
img <- to_image(train[1, px])
dim(img)
image(t(img[28:1, ]), col = grey.colors(256, start = 0, end = 1), axes = FALSE,
      main = paste("Imagen de entrenamiento 1, etiqueta", train$label[1]))
X_train <- aperm(array(t(as.matrix(train[, px])) / 255, dim = c(28, 28, nrow(train))), c(2, 1, 3))
dim(X_train)                                   # una pila de imágenes: un tensor de orden 3

# 3. Pooling: media de bloques 4 x 4, y cada imagen pasa a 7 x 7 = 49 números ---
pool7 <- function(row) {
  m <- to_image(row)
  sapply(0:48, function(k) mean(m[(k %/% 7) * 4 + 1:4, (k %% 7) * 4 + 1:4]))
}
P_train <- t(apply(train[, px], 1, pool7))
P_test  <- t(apply(test[, px], 1, pool7))
mu <- colMeans(P_train); sdv <- apply(P_train, 2, sd)   # estandarizar con las estadísticas de ENTRENAMIENTO
P_train <- scale(P_train, mu, sdv); P_test <- scale(P_test, mu, sdv)
dim(P_train)

# 4. Una neurona (regresión logística) frente a una capa oculta ---------------
library(nnet)
set.seed(2026)
neuron <- nnet(P_train, train$label, size = 0, skip = TRUE, entropy = TRUE,
               decay = 0.1, maxit = 500, trace = FALSE)
set.seed(2026)
mlp <- nnet(P_train, train$label, size = 4, entropy = TRUE,
            decay = 0.1, maxit = 500, trace = FALSE)
p_neuron <- as.numeric(predict(neuron, P_test))
p_mlp    <- as.numeric(predict(mlp, P_test))
p_cnn    <- pred$p_cnn[test$orig_index + 1]   # la CNN entrenada con las 4708 imágenes

accuracy <- function(p, y, t = 0.5) mean((p >= t) == y)
round(c(neuron = accuracy(p_neuron, test$label),
        hidden_layer = accuracy(p_mlp, test$label),
        cnn_full_data = accuracy(p_cnn, test$label)), 3)

# 5. ¿Qué errores? -------------------------------------------------------------
table(truth = test$label, predicted = as.integer(p_cnn >= 0.5))
table(truth = test$label, predicted = as.integer(p_cnn >= 0.3))
