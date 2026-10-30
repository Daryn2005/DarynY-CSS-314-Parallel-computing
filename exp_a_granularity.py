import sys
import time
from multiprocessing import Pool


def collatz_steps(n):
    steps = 0
    while n > 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def run_granularity(workers, total_items=1_000_000, chunk_size=1):
    numbers = range(1, total_items + 1)
    t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        # chunk_size=1 creates one IPC task envelope per number (micro-tasks)
        results = pool.map(collatz_steps, numbers, chunksize=chunk_size)
    return time.perf_counter() - t0, sum(results)


def karp_flatt(t1, tp, p):
    return ((p * tp / t1) - 1.0) / (p - 1.0)


if __name__ == "__main__":
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    print(f"total_items={total}")
    for label, chunk in [("micro (k=1)", 1), ("coarse (k=10000)", 10_000)]:
        print(f"\n--- {label} ---")
        times = {}
        for p in [1, 2, 4, 8]:
            t, checksum = run_granularity(p, total, chunk)
            times[p] = t
            print(f"P={p} T={t:.3f}s checksum={checksum}", flush=True)
        for p in [2, 4, 8]:
            sp = times[1] / times[p]
            print(f"P={p} Speedup={sp:.2f}x Eff={sp / p * 100:.1f}% "
                  f"KF e={karp_flatt(times[1], times[p], p):.3f}")
