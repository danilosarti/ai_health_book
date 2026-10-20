# Laboratorio del capitolo 14
# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute
# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.
# 1. Un bandito a tre solleciti per HomeBeat ------------------------------------
import numpy as np
import pandas as pd
rng = np.random.default_rng(14)
p_true = np.array([0.20, 0.35, 0.30])          # respirazione, passeggiata, sonno: ignoti agli agenti

def run_bandit(strategy, T=500, eps=0.1):
    k = len(p_true); n = np.zeros(k); s = np.zeros(k); regret = 0.0
    for t in range(T):
        if t < k:
            arm = t                                              # prova ogni sollecito una volta
        elif strategy == "thompson":
            arm = int(np.argmax(rng.beta(1 + s, 1 + n - s)))
        elif strategy == "epsilon" and rng.random() < eps:
            arm = int(rng.integers(k))
        else:
            arm = int(np.argmax(s / n))                          # scelta greedy
        y = rng.binomial(1, p_true[arm])
        n[arm] += 1; s[arm] += y
        regret += p_true.max() - p_true[arm]
    return regret, n[1] / T

print(pd.DataFrame({st: np.mean([run_bandit(st) for _ in range(50)], axis=0)
                    for st in ["greedy", "epsilon", "thompson"]},
                   index=["regret", "share_walk"]).round(3))

# 2. Il problema giocattolo dell'insulina come processo decisionale di Markov ----
bands = ["<4", "4-6", "6-10", "10-14", ">14"]
rates = ["off", "low", "medium", "high"]
acts = ["decrease", "keep", "increase"]
moves = np.array([[0.02, 0.38, 0.60],          # off:    scende, resta, sale
                  [0.10, 0.60, 0.30],          # low
                  [0.30, 0.60, 0.10],          # medium
                  [0.60, 0.38, 0.02]])         # high
reward = np.array([-10, 0, 1, 0, -1])          # per l'arrivo in ciascuna fascia
gamma = 0.9
S, A = 20, 3
band_of = np.tile(np.arange(5), 4); rate_of = np.repeat(np.arange(4), 5)   # s = velocità * 5 + fascia
P = np.zeros((S, A, S))
for s in range(S):
    for a in range(A):
        r2 = min(3, max(0, rate_of[s] + a - 1))
        for m in range(3):
            g2 = min(4, max(0, band_of[s] + m - 1))
            P[s, a, r2 * 5 + g2] += moves[r2, m]
R = P @ reward[band_of]                         # ricompensa attesa di ogni (s, a), forma (S, A)

# 3. Iterazione dei valori: applicare l'equazione di ottimalità di Bellman fino a stabilità
def value_iteration(R, sweeps=500):
    V = np.zeros(S)
    for _ in range(sweeps):
        Q = R + gamma * P @ V
        V = Q.max(axis=1)
    return Q, V, Q.argmax(axis=1)

Qopt, Vopt, policy = value_iteration(R)                          # (con 'off', decrease = keep)
print(pd.DataFrame(np.array(acts)[policy].reshape(4, 5).T, index=bands, columns=rates))
print(pd.DataFrame(Vopt.reshape(4, 5).T, index=bands, columns=rates).round(2))

# 4. Q-learning da esperienza simulata --------------------------------------------
rng = np.random.default_rng(2014)
Qhat = np.zeros((S, A)); eta, eps = 0.1, 0.2
for episode in range(3000):
    s = int(rng.integers(S))                                     # stato iniziale casuale
    for t in range(12):                                          # 24 ore, passi di 2 ore
        a = int(rng.integers(A)) if rng.random() < eps else int(np.argmax(Qhat[s]))
        s2 = int(rng.choice(S, p=P[s, a]))
        r = reward[band_of[s2]]
        Qhat[s, a] += eta * (r + gamma * Qhat[s2].max() - Qhat[s, a])
        s = s2
learned = Qhat.argmax(axis=1)
print("quota quasi ottima:", np.mean(Qopt[np.arange(S), learned] >= Vopt - 0.05))

# 5. Confrontare le politiche su 24 ore, in modo esatto ----------------------------
def evaluate(policy, start=3):                                   # glicemia 10-14, infusione spenta
    d = np.zeros(S); d[start] = 1; out = np.zeros(3)
    Ppi = P[np.arange(S), policy]
    for t in range(12):
        d = d @ Ppi
        out += [d[band_of == 2].sum(), d[band_of == 0].sum(), d[band_of >= 3].sum()]
    return pd.Series(out / 12, index=["in_range", "hypo", "hyper"]).round(3)

sliding = np.where(band_of >= 3, 2, np.where(band_of <= 1, 0, 1))   # schema a scalare al letto del paziente
print(pd.DataFrame({"sliding_scale": evaluate(sliding), "optimal": evaluate(policy),
                    "q_learning": evaluate(learned)}).T)

# 6. Una ricompensa mal specificata: '+1 ogni volta che la glicemia è sotto 10' ----
R_bad = P @ np.array([1, 1, 1, 0, 0])[band_of]
print("ricompensa sotto 10:", evaluate(value_iteration(R_bad)[2]).to_dict())

# 7. Q-learning offline dalle cartelle di cure con schema a scalare -----------------
rng = np.random.default_rng(7)
D = []                                                           # righe: s, a, r, s'
for episode in range(500):
    s = int(rng.integers(2, 5))                                  # ricoverato con 6-14+, infusione spenta
    for t in range(12):
        a = sliding[s]; s2 = int(rng.choice(S, p=P[s, a]))
        D.append((s, a, reward[band_of[s2]], s2)); s = s2
Qoff = np.zeros((S, A))
for sweep in range(20):
    for j in rng.permutation(len(D)):
        s, a, r, s2 = D[j]
        Qoff[s, a] += 0.1 * (r + gamma * Qoff[s2].max() - Qoff[s, a])
print("quota mai provata:", 1 - len({(s, a) for s, a, _, _ in D}) / (S * A))
print(pd.DataFrame({"nurses_in_records": evaluate(sliding), "offline_agent": evaluate(Qoff.argmax(axis=1))}).T)
