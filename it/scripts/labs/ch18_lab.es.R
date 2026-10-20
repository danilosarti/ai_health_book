# Laboratorio del capítulo 18
# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud
# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.
# 1. Cargar los fragmentos de las guías y construir un índice TF-IDF ------------------------
guidelines <- read.csv("data/ch18_guidelines.csv", encoding = "UTF-8")
stop_words <- c("a","an","and","are","as","at","be","by","can","do","does","for","from","has","have",
  "how","i","if","in","is","it","its","not","of","on","or","our","should","so","that","the","their",
  "them","then","there","these","they","this","to","was","we","what","when","which","while","who",
  "will","with","without","you","been","into","than","after","before","more","most","also","all",
  "any","every","each","may","only","other","such","whenever","where","one","two")
tokens <- function(x) {
  w <- unlist(strsplit(tolower(x), "[^a-z0-9]+"))
  w[nchar(w) >= 2 & !(w %in% stop_words)]
}
docs  <- lapply(paste(guidelines$section, guidelines$text), tokens)
vocab <- sort(unique(unlist(docs)))
tf    <- t(sapply(docs, function(d) table(factor(d, levels = vocab))))
idf   <- log((1 + nrow(tf)) / (1 + colSums(tf > 0))) + 1
X     <- sweep(tf, 2, idf, "*")
X     <- X / sqrt(rowSums(X^2))
embed <- function(q) {
  v <- as.numeric(table(factor(tokens(q), levels = vocab))) * idf
  if (sum(v^2) > 0) v / sqrt(sum(v^2)) else v
}
retrieve <- function(q, k = 3) {
  s <- as.vector(X %*% embed(q)); o <- order(-s)[1:k]
  data.frame(chunk_id = guidelines$chunk_id[o], score = round(s[o], 3), text = guidelines$text[o])
}
c(chunks = nrow(X), terms = ncol(X))

# 2. Recuperar fragmentos para la pregunta de Seán ------------------------------------------------
question <- paste("Mrs Byrne is on vancomycin and piperacillin-tazobactam, and her creatinine rose from",
                  "1.0 to 1.4 mg/dL in 36 hours. Should we change the vancomycin, and how often should",
                  "we check creatinine?")
hits <- retrieve(question, k = 3)
hits[, c("chunk_id", "score")]

# 3. Montar el prompt fundamentado (esto es lo que se enviaría al modelo) ------
prompt <- paste(c(
  "Eres el asistente de guías de la UCI Sala 7. Responde SOLO a partir de los fragmentos numerados de abajo.",
  "Cita el identificador del fragmento en cada afirmación. Si los fragmentos no contienen la respuesta, dilo.",
  "", paste0("[", hits$chunk_id, "] ", hits$text), "", paste("Pregunta:", question)), collapse = "\n")
cat(prompt, "\n")

# 4. Comprobación rudimentaria de la fundamentación de una respuesta fija ----------------------------------
# Cada frase se compara con los fragmentos recuperados; similitud baja = señalar para revisión.
answer <- c("Do not stop vancomycin without discussing it with the microbiology team.",
            "Consider an alternative beta-lactam to piperacillin-tazobactam and monitor creatinine daily.",
            "Mrs Byrne's risk of needing dialysis is about 10 percent.")
Xhits   <- X[match(hits$chunk_id, guidelines$chunk_id), , drop = FALSE]
support <- sapply(answer, function(s) max(Xhits %*% embed(s)))
data.frame(sentence = substr(answer, 1, 45), best_support = round(support, 2), flag = support < 0.2,
           row.names = NULL)

# 5. Evaluación: exactitud y alucinación por configuración ---------------------
ev <- read.csv("data/ch18_eval.csv")
wilson <- function(k, n, z = 1.96) {
  p <- k / n; c0 <- (p + z^2 / (2 * n)) / (1 + z^2 / n)
  h <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / (1 + z^2 / n); c(c0 - h, c0 + h)
}
summ <- do.call(rbind, lapply(split(ev, ev$config), function(d) {
  ci <- wilson(sum(d$hallucinated), nrow(d))
  data.frame(config = d$config[1], accuracy = mean(d$correct), hallucinated = mean(d$hallucinated),
             hall_lo = ci[1], hall_hi = ci[2], abstained = mean(d$abstained))
}))
print(summ, digits = 3, row.names = FALSE)
round(tapply(ev$correct, list(ev$category, ev$config), mean), 2)

# 6. Calibración y respuesta selectiva -----------------------------------------
tapply(ev$stated_conf, ev$config, mean) - tapply(ev$correct, ev$config, mean)   # exceso de confianza
rag <- ev[ev$config == "rag", ]
thr <- round(seq(0.5, 0.95, by = 0.05), 2)
data.frame(threshold = thr,
           coverage  = sapply(thr, function(t) mean(rag$stated_conf >= t)),
           accuracy  = sapply(thr, function(t) mean(rag$correct[rag$stated_conf >= t])))

# 7. Rentosertib, fase 2a: comparación aproximada a partir de los intervalos publicados --------
ci60 <- c(10.9, 185.9); cipl <- c(-116.1, 75.6)        # cambio de la CVF a las 12 semanas (mL)
se   <- c(diff(ci60), diff(cipl)) / (2 * 1.96)
diff_fvc <- mean(ci60) - mean(cipl); se_diff <- sqrt(sum(se^2))
round(c(difference = diff_fvc, lower = diff_fvc - 1.96 * se_diff, upper = diff_fvc + 1.96 * se_diff), 1)
