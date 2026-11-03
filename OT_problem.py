import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import logsumexp


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

    logK = -beta * C
    loga = np.zeros(n)
    logb = np.zeros(m)

    for t in range(T):
        loga = np.log(p) - logsumexp(logK + logb[None, :], axis=1)
        logb = np.log(q) - logsumexp(logK + loga[:, None], axis=0)

    pi = np.exp(loga[:, None] + logK + logb[None, :])

    return(pi)




## ANNEALED SINKHORN ALGORITHM
def annealed_sinkhorn(C, p, q, beta0, kappa, T=1000):
    beta = beta0 * (1 + np.arange(T + 1))**kappa

    logK = -beta[0] * C
    loga = np.zeros(len(p))
    logb = np.zeros(len(q))

    for t in range(1, T + 1):
        loga = np.log(p) - logsumexp(logK + logb[None, :], axis=1)
        logK = -beta[t] * C
        logb = np.log(q) - logsumexp(logK + loga[:, None], axis=0)

    pi = np.exp(loga[:, None] + logK + logb[None, :])
    return(pi)




## DEBIAISED ANNEALED SINKHORN ALGORITHM


def debiased_annealed_sinkhorn(C, p, q, beta0, kappa=2/3, T=1000):
    beta = beta0 * (1 + np.arange(T + 1))**kappa
    logK = -beta[0] * C
    loga = np.zeros(len(p))
    logb = np.zeros(len(q))

    for t in range(1, T + 1):
        exponent = 1 - beta[max(0, t-2)] / beta[t-1]
        loga = np.log(p) + exponent * loga - logsumexp(logK + logb[None, :], axis=1)
        logK = -beta[t] * C
        logb = np.log(q) - logsumexp(logK + loga[:, None], axis=0)
    pi = np.exp(loga[:, None] + logK + logb[None, :])
    return(pi)


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


betas = [0.1,0.5, 1, 2, 5, 10, 20,50]
errors = []
times = []

for beta in betas:
    start = time.perf_counter()
    pi = sinkhorn(C, p, q, beta, T=1000)
    times.append(time.perf_counter() - start)

    errors.append(np.sum(np.abs(pi - pi_opt)))

plt.figure()
plt.semilogy(betas, errors, "o-")
plt.grid(alpha=0.5,which="both")
plt.xlabel("Beta")
plt.ylabel("Distance to exact OT")
plt.title("Sinkhorn: influence of beta")
plt.savefig("sinkhorn_beta.png", dpi=300, bbox_inches="tight")
plt.close()


## STUDY ANNEALED SINKHORN

kappas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

errors_annealed = []
errors_debiased = []

for kappa in kappas:
    pi_a = annealed_sinkhorn(C, p, q, beta0=0.1, kappa=kappa, T=1000)
    pi_d = debiased_annealed_sinkhorn(C, p, q, beta0=0.1, kappa=kappa, T=1000)

    errors_annealed.append(np.sum(np.abs(pi_a - pi_opt)))
    errors_debiased.append(np.sum(np.abs(pi_d - pi_opt)))

plt.figure()
plt.plot(kappas, errors_annealed, "o-", label="Annealed")
plt.plot(kappas, errors_debiased, "o-", label="Debiased")
plt.xlabel("Kappa")
plt.ylabel("Distance to exact OT")
plt.grid(alpha=0.5,which="both")
plt.title("Influence of kappa")
plt.legend()
plt.savefig("annealed_kappa.png", dpi=300, bbox_inches="tight")
plt.close()

## STUDY BETA0

beta0_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1]

errors_annealed = []
errors_debiased = []

for beta0 in beta0_values:
    pi_a = annealed_sinkhorn(C, p, q, beta0, kappa=0.5, T=1000)
    pi_d = debiased_annealed_sinkhorn(C, p, q, beta0, kappa=0.5, T=1000)

    errors_annealed.append(np.sum(np.abs(pi_a - pi_opt)))
    errors_debiased.append(np.sum(np.abs(pi_d - pi_opt)))

plt.figure()
plt.plot(beta0_values, errors_annealed, "o-", label="Annealed")
plt.plot(beta0_values, errors_debiased, "o-", label="Debiased")
plt.xlabel("Beta0")
plt.ylabel("Distance to exact OT")
plt.grid(alpha=0.5,which="both")
plt.title("Influence of beta0")
plt.legend()
plt.savefig("annealed_beta0.png", dpi=300, bbox_inches="tight")
plt.close()

## COMPARISON

steps = [10, 20, 50, 100, 200, 500, 1000]

plt.figure()

for method, name in [
    (lambda T: sinkhorn(C, p, q, beta=10, T=T), "Sinkhorn"),
    (lambda T: annealed_sinkhorn(C, p, q, 0.1, 0.5, T), "Annealed"),
    (lambda T: debiased_annealed_sinkhorn(C, p, q, 0.1, 2/3, T), "Debiased")
]:
    errors = []
    times = []

    for T in steps:
        start = time.perf_counter()
        pi = method(T)
        times.append(time.perf_counter() - start)

        errors.append(np.sum(np.abs(pi - pi_opt)))

    plt.plot(times, errors, "o-", label=name)

plt.xlabel("Time (s)")
plt.ylabel("Distance to exact OT")
plt.grid(alpha=0.5,which="both")
plt.title("Comparison with exact OT")
plt.legend()
plt.savefig("comparison_time.png", dpi=300, bbox_inches="tight")
plt.close()


## MARGINAL ERROR: ANNEALED VS DEBIASED

kappas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

errors_annealed = []
errors_debiased = []

for kappa in kappas:
    pi_a = annealed_sinkhorn(C, p, q, 0.1, kappa, T=1000)
    pi_d = debiased_annealed_sinkhorn(C, p, q, 0.1, kappa, T=1000)

    errors_annealed.append(
        np.sum(np.abs(pi_a.sum(axis=1) - p))
    )
    errors_debiased.append(
        np.sum(np.abs(pi_d.sum(axis=1) - p))
    )

plt.figure()
plt.semilogy(kappas, errors_annealed, "o-", label="Annealed")
plt.semilogy(kappas, errors_debiased, "o-", label="Debiased")
plt.xlabel("Kappa")
plt.ylabel("Marginal error")
plt.title("Relaxation error")
plt.grid(alpha=0.5, which="both")
plt.legend()
plt.savefig("marginal_error.png", dpi=300, bbox_inches="tight")
plt.close()



steps = [10, 20, 50, 100, 200, 500, 1000, 2000]

plt.figure()

for kappa in [0.5, 2/3]:
    errors_a = []
    errors_d = []

    for T in steps:
        pi_a = annealed_sinkhorn(C, p, q, 0.1, kappa, T)
        pi_d = debiased_annealed_sinkhorn(C, p, q, 0.1, kappa, T)

        errors_a.append(np.sum(np.abs(pi_a.sum(axis=1) - p)))
        errors_d.append(np.sum(np.abs(pi_d.sum(axis=1) - p)))

    plt.loglog(steps, errors_a, "o-", label=f"Annealed, k={kappa:.2f}")
    plt.loglog(steps, errors_d, "o--", label=f"Debiased, k={kappa:.2f}")

plt.xlabel("Iterations")
plt.ylabel("First marginal error")
plt.title("Marginal convergence")
plt.grid(alpha=0.5, which="both")
plt.legend()
plt.savefig("marginal_convergence.png", dpi=300, bbox_inches="tight")
plt.close()


for T in [500, 1000, 2000]:
    pi = debiased_annealed_sinkhorn(C, p, q, 0.1, 2/3, T)

    print("T =", T)
    print("Finite:", np.isfinite(pi).all())
    print("Mass:", np.sum(pi))
    print("Marginal error:", np.sum(np.abs(pi.sum(axis=1) - p)))