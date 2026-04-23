import numpy as np
from scipy.stats import ttest_rel

def parse_mu_sigma(part: str):
    part = part.strip()
    if "±" not in part:
        raise ValueError(f"Bad format (missing ±): {part}")
    mu_str, sigma_str = part.split("±", 1)
    mu = float(mu_str.strip())
    sigma = float(sigma_str.strip())
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    return mu, sigma

def run_paired_ttest_from_str(s: str, n: int = 100, seed: int = 42):
    # input format: "0.11±0.006$$$0.17±0.004"
    if "$$$" not in s:
        raise ValueError("Bad format (missing $$$ separator). Expect: A$$$B")
    a_str, b_str = s.split("$$$", 1)

    mu_a, sigma_a = parse_mu_sigma(a_str)
    mu_b, sigma_b = parse_mu_sigma(b_str)

    rng = np.random.default_rng(seed)
    A = rng.normal(loc=mu_a, scale=sigma_a, size=n)
    B = rng.normal(loc=mu_b, scale=sigma_b, size=n)

    t_stat, p_val = ttest_rel(A, B)

    return {
        "A_mu": mu_a, "A_sigma": sigma_a,
        "B_mu": mu_b, "B_sigma": sigma_b,
        "n": n, "seed": seed,
        "t_stat": float(t_stat),
        "p_val": float(p_val),
        "A_sample_mean": float(A.mean()),
        "B_sample_mean": float(B.mean()),
        "A": A,
        "B": B,
    }

if __name__ == "__main__":
    inp = "0.53±0.016$$$0.69±0.020"
    res = run_paired_ttest_from_str(inp, n=5, seed=42)
    print("t_stat:", res["t_stat"], "p_val:", res["p_val"])
    print("A mean (sample):", res["A_sample_mean"], "B mean (sample):", res["B_sample_mean"])