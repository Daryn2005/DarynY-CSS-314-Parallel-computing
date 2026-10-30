# RESULTS_B: Scalability Diagnostics (Variant B)

**Hardware:** shared cloud sandbox with **2 CPUs** (`os.cpu_count() = 2`). Results for P = 4 and P = 8 are oversubscribed, so they cannot reach the handout's expected 8-core numbers. Runs were done sequentially, one trial each (no averaging), so expect noise.

**Workloads:** Exp A = Collatz on 200,000 numbers (reduced from 1,000,000 for time). Exp B = 400,000 queue items. Exp C = 10,000,000 items, batch size 10,000.

## Main table

| Configuration | T1 (s) | T2 (s) | T4 (s) | T8 (s) | S8 | e(2) | e(8) | Verdict |
|---|---|---|---|---|---|---|---|---|
| A: Micro (k=1) | 8.403 | 5.400 | 5.529 | 6.960 | 1.21 | 0.285 | 0.804 | Master dispatch/pickling overhead dominates |
| A: Coarse (k=10,000) | 1.537 | 0.792 | 0.866 | 0.918 | 1.68 | 0.031 | 0.539 | Healthy at P=2; P>2 limited by 2 physical CPUs |
| B: Queue choke | 3.623 | 2.965 | 2.641 | 2.041 | 1.77 | 0.637 | 0.501 | High e at all P (IPC cost); negative scaling not reproduced here |
| C: Buffered batch | 0.911 | 0.321 | 0.318 | 0.310 | 2.94 | -0.296 | 0.246 | Amortized synchronization; speedup capped by 2 CPUs |

## Karp-Flatt detail (e = ((p*Tp/T1) - 1) / (p - 1))

| Config | e(2) | e(4) | e(8) |
|---|---|---|---|
| A micro | 0.285 | 0.544 | 0.804 |
| A coarse | 0.031 | 0.418 | 0.539 |
| B queue | 0.637 | 0.639 | 0.501 |
| C buffered | -0.296 | 0.132 | 0.246 |

## Findings

**Experiment A (granularity).** At P=1, k=1 takes 8.40 s versus 1.54 s for k=10,000. Same computation, same checksum (22,938,602), about 5.5x slower, purely from per-task pickling and dispatch (200,000 task envelopes). Micro-tasks scale poorly (S2 = 1.56x, S8 = 1.21x) and e(p) rises steadily with p (0.285 -> 0.544 -> 0.804), the signature of master-process dispatch serialization. Coarse chunks reach S2 = 1.94x (97% efficiency, e = 0.031). Beyond P=2 the coarse runs flatten because only 2 CPUs exist.

**Experiment B (queue).** The queue is slow at every core count: P=1 manages about 110k items/s. Karp-Flatt stays high (0.50-0.64), meaning large non-parallel overhead from the pipe, locks and the single consumer. However, **the handout's predicted negative scaling (T8 > T1, e(8) > 2.0) was NOT reproduced**: T8 (2.04 s) < T1 (3.62 s). Likely reasons: in this design the master is the consumer bottleneck, and on 2 CPUs the extra producers mostly add concurrency around it; the handout's example numbers also look like illustrative values, not measurements. e(p) is not monotonically increasing here (0.637, 0.639, 0.501).

**Experiment C (buffered).** Batching 10,000 items per put cuts IPC to ~1,000 messages total. Even though it processes 25x more data (10M vs 400k), it finishes in 0.31-0.91 s versus 2.0-3.6 s for B. Throughput at P=8: about 32M items/s vs about 196k items/s, roughly **165x**. Speedup is capped near 2-3x by the 2 CPUs, so S8 > 5.0 and e(8) <= 0.05 are not achievable on this hardware. e(2) = -0.296 (superlinear) is a measurement artifact: the single P=1 run was likely slowed by process start-up/CPU-frequency noise; repeat several times and take the median to confirm.

Note: the C checksums differ slightly across P (29999994, ...90, ...88, ...52) because each worker's `i % 7` sequence restarts at 0 and items-per-worker is not a multiple of 7. That is expected, not a bug.

## Sign-off criteria check

| Criterion | Result |
|---|---|
| A: chunksize=1 shows dispatch starvation | Met (5.5x overhead at P=1, e(8) = 0.80) |
| B: T8 > T1 and monotonic e(p) | Not reproduced on 2 CPUs |
| C: > 10x throughput | Met (~100-165x) |
| C: S8 > 5.0 and e(8) <= 0.05 | Not achievable on 2 CPUs (S8 = 2.94, e(8) = 0.246) |

To meet the B and C targets, rerun on a machine with 8+ physical cores, 3-5 trials per point, and report the median.
