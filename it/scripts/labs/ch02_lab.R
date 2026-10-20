# Chapter 2 Lab
# AI and Machine Learning for Health Sciences
# Run from the book folder (the one that contains data/), or change the paths.
# 1. Load the data ----------------------------------------------------------
icu <- read.csv("data/icu_aki.csv")
hb  <- read.csv("data/homebeat.csv")
icu$aki <- icu$delta_creatinine_48h >= 0.3       # KDIGO criterion: a Bernoulli outcome

# 2. Bernoulli and binomial -------------------------------------------------
theta <- mean(icu$aki)                            # MLE of the probability of AKI
theta
dbinom(0:20, size = 20, prob = theta)             # distribution of AKI cases on a ward of 20
1 - pbinom(14, size = 20, prob = theta)           # P(15 or more cases)

# 3. Normal model for resting heart rate ------------------------------------
mu <- mean(hb$resting_hr); s <- sd(hb$resting_hr)
c(model = 1 - pnorm(80, mu, s), data = mean(hb$resting_hr > 80))   # P(HR > 80)
hist(hb$resting_hr, breaks = 30, freq = FALSE, main = "", xlab = "Resting heart rate (bpm)")
curve(dnorm(x, mu, s), add = TRUE, lwd = 3)

# 4. Likelihood and MLE on a grid -------------------------------------------
theta_grid <- seq(0.01, 0.99, by = 0.001)
loglik <- sapply(theta_grid, function(th) sum(dbinom(icu$aki, 1, th, log = TRUE)))
theta_grid[which.max(loglik)]                    # equals mean(icu$aki), up to the grid
plot(theta_grid, exp(loglik - max(loglik)), type = "l", lwd = 2,
     xlab = "theta", ylab = "Likelihood (peak = 1)")

# 5. Bayes by counting: frailty given a fast heart rate (HomeBeat) ----------
frail <- hb$frailty >= 0.5; fast <- hb$resting_hr > 85
table(frail, fast)
mean(frail[fast])                                 # posterior, counted directly
pF <- mean(frail); a <- mean(fast[frail]); b <- mean(fast[!frail])
a * pF / (a * pF + b * (1 - pF))                  # posterior, by Bayes' theorem

# 6. A diagnostic test for tonight's patients -------------------------------
se <- 0.85; sp <- 0.80
prior <- c(Tomas = 0.07, Aoife = 0.49, Priya = 0.99)   # risks from the Chapter 5 model
p_pos    <- prior * se + (1 - prior) * (1 - sp)
post_pos <- prior * se / p_pos
post_neg <- prior * (1 - se) / (1 - p_pos)

# 7. Expected loss, threshold and value of information ----------------------
C_FP <- 10; C_FN <- 30
t_star <- C_FP / (C_FP + C_FN)
best <- function(q) pmin((1 - q) * C_FP, q * C_FN)  # expected loss of the best action
loss_no_test   <- best(prior)
loss_with_test <- p_pos * best(post_pos) + (1 - p_pos) * best(post_neg)
round(data.frame(prior, post_pos, post_neg, loss_no_test, loss_with_test,
                 value_of_test = loss_no_test - loss_with_test), 3)

# 8. Entropy -----------------------------------------------------------------
H <- function(q) -q * log2(q) - (1 - q) * log2(1 - q)
round(H(prior), 3)
