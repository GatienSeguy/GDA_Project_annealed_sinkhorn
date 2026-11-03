import numpy as np
import matplotlib.pyplot as plt
import time

### Generate random 2D OT problem
def generate_OT_problem(n,m):
    X = np.random.normal(size=(n,2))
    Y = np.random.normal(size=(m,2))
    p = np.ones(n)/n
    q = np.ones(m)/m
    return(X,Y,p,q)

def cost_function(x,y):
    return(np.linalg.norm(x-y)**2)

def compute_cost_matrix(X,Y,c):
    n = X.shape[0]
    m= Y.shape[0]
    C = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            C[i,j] = c(X[i],Y[j])
    return(C)


### Optimal Transport Problem (O(n3) algorithm)
def optimal_transport(C):
    N = len(C)

    u = np.zeros(N + 1)
    v = np.zeros(N + 1)
    match = np.zeros(N + 1, dtype=int)

    for i in range(1, N + 1):
        match[0] = i
        best = np.full(N + 1, np.inf)
        prev = np.zeros(N + 1, dtype=int)
        used = np.zeros(N + 1, dtype=bool)
        j0 = 0

        while True:
            used[j0] = True
            row = match[j0]
            delta = np.inf

            for j in range(1, N + 1):
                if not used[j]:
                    cost = C[row-1, j-1] - u[row] - v[j]

                    if cost < best[j]:
                        best[j] = cost
                        prev[j] = j0

                    if best[j] < delta:
                        delta = best[j]
                        j1 = j

            for j in range(N + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    best[j] -= delta

            j0 = j1
            if match[j0] == 0:
                break
        while j0 != 0:
            j1 = prev[j0]
            match[j0] = match[j1]
            j0 = j1
    pi = np.zeros((N, N))
    for j in range(1, N + 1):
        pi[match[j]-1, j-1] = 1 / N
    return(pi)


## SINKHORN ALGORITHM
def sinkhorn(C, p, q, beta, T=1000):
    n, m = C.shape
    K = np.exp(-beta*C)
    a = np.ones(n)
    b = np.ones(m)

    for i in range(T):
        a = p / (K @ a)
        a = q / (K.T @ b)

    pi = np.diag(a) @ K @ np.diag(b)
    return(pi)



## ANNEALED SINKHORN ALGORITHM
def annealed_sinkhorn(C, p, q, beta0, kappa, T=1000):
    beta = beta0 * (1 + np.arange(T + 1))**kappa


    K = np.exp(-beta[0] * C)
    b = np.ones(len(q))

    for t in range(1, T + 1):
        a = p / (K @ b)
        K = np.exp(-beta[t] * C)
        b = q / (K.T @ a)

    pi = np.diag(a) @ K @ np.diag(b)
    return(pi)



## DEBIAISED ANNEALED SINKHORN ALGORITHM

def debiased_annealed_sinkhorn(C, p, q, beta0, kappa=2/3, T=1000):
    beta = beta0 * (1 + np.arange(T + 1))**kappa

    K = np.exp(-beta[0] * C)
    a = np.ones(len(p))
    b = np.ones(len(q))

    for t in range(1, T + 1):
        exponent = 1 - beta[max(0, t-2)] / beta[t-1]

        a = (p * a**exponent) / (K @ b)
        K = np.exp(-beta[t] * C)
        b = q / (K.T @ a)

    pi = a[:, None] * K * b[None, :]

    return pi


# TEST
N = 256
X, Y, p, q = generate_OT_problem(N, N)
C = compute_cost_matrix(X, Y, cost_function)

# Exact OT
start = time.perf_counter()
pi_opt = optimal_transport(C)
time_opt = time.perf_counter() - start

# Sinkhorn
start = time.perf_counter()
pi_sinkhorn = sinkhorn(C, p, q, beta=2, T=1000)
time_sinkhorn = time.perf_counter() - start

# Annealed Sinkhorn
start = time.perf_counter()
pi_annealed = annealed_sinkhorn(C, p, q, beta0=0.1, kappa=0.5, T=1000)
time_annealed = time.perf_counter() - start

# Debiased Annealed Sinkhorn
start = time.perf_counter()
pi_debiased = debiased_annealed_sinkhorn(C, p, q, beta0=0.1, kappa=0.5, T=1000)
time_debiased = time.perf_counter() - start

# Results
methods = [
    ("Exact OT", pi_opt, time_opt),
    ("Sinkhorn", pi_sinkhorn, time_sinkhorn),
    ("Annealed Sinkhorn", pi_annealed, time_annealed),
    ("Debiased Annealed", pi_debiased, time_debiased)
]

for name, pi, duration in methods:
    cost = np.sum(C * pi)
    error_p = np.sum(np.abs(pi.sum(axis=1) - p))
    error_q = np.sum(np.abs(pi.sum(axis=0) - q))
    distance = np.sum(np.abs(pi - pi_opt))

    print(f"\n{name}")
    print(f"  Time: {duration:.4f} s")
    print(f"  Transport cost: {cost:.6f}")
    print(f"  Distance to OT plan: {distance:.6f}")
    print(f"  Marginal error p: {error_p:.6f}")
    print(f"  Marginal error q: {error_q:.6f}")







##### COMPARAISON PLOTS

