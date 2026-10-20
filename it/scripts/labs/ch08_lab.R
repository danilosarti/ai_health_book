# Chapter 8 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Conjugate learning of the ICU_D AKI rate ------------------------------------
icu <- read.csv("data/icu_aki.csv")
akiD <- icu$delta_creatinine_48h[icu$site == "ICU_D"] >= 0.3
a0 <- 55; b0 <- 45                                    # prior from the other hospitals
for (n in c(0, 10, 50, 313)) {
  k <- sum(akiD[seq_len(n)])
  ci <- qbeta(c(0.025, 0.975), a0 + k, b0 + n - k)
  cat(sprintf("n = %3d  k = %3d  posterior mean %.3f  95%% CrI %.3f-%.3f\n",
              n, k, (a0 + k) / (a0 + b0 + n), ci[1], ci[2]))
}

# 2. Grid posterior of the HomeBeat frailty slope (intercept and sigma held fixed) -
hb <- read.csv("data/homebeat.csv")
x <- hb$frailty - mean(hb$frailty); y <- hb$resting_hr
grid <- seq(20, 70, by = 0.05)
logpost <- sapply(grid, function(b1) sum(dnorm(y, mean(y) + b1 * x, 11.2, log = TRUE))) +
  dnorm(grid, 0, 20, log = TRUE)
post <- exp(logpost - max(logpost)); post <- post / sum(post)
c(posterior_mean = sum(grid * post), least_squares = unname(coef(lm(y ~ x))[2]))

# 3. Stan draws: summaries and the confounding lesson -------------------------------
simple   <- read.csv("data/precomputed/ch08/homebeat_simple_draws.csv")
adjusted <- read.csv("data/precomputed/ch08/homebeat_adjusted_draws.csv")
round(tapply(simple$b_frailty, simple$chain, mean), 2)    # do the two chains agree?
rbind(simple   = c(quantile(simple$b_frailty, c(0.025, 0.5, 0.975)), p_pos = mean(simple$b_frailty > 0)),
      adjusted = c(quantile(adjusted$b_frailty, c(0.025, 0.5, 0.975)), p_pos = mean(adjusted$b_frailty > 0)))

# 4. Lena: credible interval of the mean versus prediction interval -----------------
mu <- with(adjusted, b0 + b_frailty * 0.55 + b_age * 72 + b_infl * 4.1)
set.seed(8)
y_new <- rnorm(length(mu), mu, adjusted$sigma)
rbind(average_of_people_like_Lena = quantile(mu, c(0.025, 0.975)),
      Lenas_next_reading          = quantile(y_new, c(0.025, 0.975)))
mean(y_new > 88)                                       # probability of a reading above 88 bpm

# 5. Tonight on Ward 7: risk of AKI with uncertainty --------------------------------
risk <- read.csv("data/precomputed/ch08/ward7_tonight_risk_draws.csv")
summ <- function(r, t) c(mean = mean(r), quantile(r, c(0.025, 0.975)), p_above_t = mean(r > t))
round(t(sapply(risk, summ, t = 0.25)), 3)
round(t(sapply(risk, summ, t = 0.09)), 3)

# 6. BART versus forests and boosting (honest test-set errors) ---------------------
# BART and the other models were fitted once, outside the browser, on the shared splits;
# their test errors are read from the book's model ledger.
hbl <- read.csv("data/precomputed/ledger/homebeat.csv")
hbl[hbl$role %in% c("reference", "canonical"), c("model", "test_MAE", "test_RMSE", "test_R2")]
w7 <- read.csv("data/precomputed/ledger/ward7_reg.csv")
w7[w7$chapter <= 8 & w7$role %in% c("reference", "canonical", "teaching"),
   c("chapter", "model", "test_MAE", "test_RMSE")]
read.csv("data/precomputed/ch08/bart_comparison.csv")[, c("data", "model", "train_RMSE", "coverage95")]
