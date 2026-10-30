import time
from multiprocessing import Process, Queue


def buffered_worker(out_queue, items_per_worker, batch_size=10_000):
    buffer = []
    for i in range(items_per_worker):
        buffer.append(i % 7)
        if len(buffer) >= batch_size:
            out_queue.put(sum(buffer))  # flush a pre-aggregated chunk
            buffer.clear()
    if buffer:
        out_queue.put(sum(buffer))
    out_queue.put(None)  # sentinel end-of-work token


def run_buffered(p, total_items=10_000_000):
    q = Queue()
    items_per_worker = total_items // p
    workers = [Process(target=buffered_worker, args=(q, items_per_worker)) for _ in range(p)]
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    total_sum, active = 0, p
    while active > 0:
        val = q.get()
        if val is None:
            active -= 1
        else:
            total_sum += val
    for w in workers:
        w.join()
    return time.perf_counter() - t0, total_sum


if __name__ == "__main__":
    cores = [1, 2, 4, 8]
    times = {}
    for p in cores:
        t, s = run_buffered(p)
        times[p] = t
        print(f"P={p} T={t:.3f}s sum={s}", flush=True)
    t1 = times[1]
    print("\n--- Buffered Karp-Flatt Analysis ---")
    for p in [2, 4, 8]:
        tp = times[p]
        sp = t1 / tp
        e = ((p * tp / t1) - 1.0) / (p - 1.0)
        print(f"P={p} | Time={tp:6.3f}s | Speedup={sp:5.2f}x | Eff={sp / p * 100:5.1f}% | e={e:6.3f}")
