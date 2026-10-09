"""
CUDA Lab 02 - all tasks in one file.

Run on a GPU runtime (e.g. Colab T4):
    python cuda_lab_02_all.py

Sections:
    Task 1: warp divergence microbenchmark
    Task 2: 1D stencil with halo replication
    Task 3: grid-stride vector scaling
    Task 4: 2D Sobel-X convolution
    Verification + integrity token (same logic as verify_submission.py)
"""
import sys
import math
import time
import hashlib
import numpy as np
from numba import cuda, float32


# ===========================================================================
# Task 1: Warp divergence microbenchmark
# ===========================================================================
T1_N = 2 ** 20
T1_ITERS = 1000
T1_THREADS = 256


@cuda.jit
def kernel_a_uniform(y, n):
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        for _ in range(T1_ITERS):
            v = v * float32(1.0001) + float32(0.0001)
        y[idx] = v


@cuda.jit
def kernel_b_interleaved(y, n):
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        if idx % 2 == 0:
            for _ in range(T1_ITERS):  # Path 1: multiply-accumulate
                v = v * float32(1.0001) + float32(0.0001)
        else:
            for _ in range(T1_ITERS):  # Path 2: subtract-divide
                v = (v - float32(0.0001)) / float32(1.0001)
        y[idx] = v


@cuda.jit
def kernel_c_warp_aligned(y, n):
    idx = cuda.grid(1)
    if idx < n:
        v = y[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:
            for _ in range(T1_ITERS):  # Path 1
                v = v * float32(1.0001) + float32(0.0001)
        else:
            for _ in range(T1_ITERS):  # Path 2
                v = (v - float32(0.0001)) / float32(1.0001)
        y[idx] = v


def bench(kernel, d_y, n, trials=10):
    """Kernel-only time in ms: 1 warm-up launch, then mean of `trials` runs."""
    blocks = (n + T1_THREADS - 1) // T1_THREADS
    kernel[blocks, T1_THREADS](d_y, n)  # warm-up (also triggers JIT compile)
    cuda.synchronize()
    times = []
    for _ in range(trials):
        cuda.synchronize()
        t0 = time.perf_counter()
        kernel[blocks, T1_THREADS](d_y, n)
        cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)) * 1000.0


def run_task1():
    h_y = np.ones(T1_N, dtype=np.float32)
    d_y = cuda.to_device(h_y)  # transfer excluded from timing
    results = {}
    for name, k in [("A (Uniform)", kernel_a_uniform),
                    ("B (Interleaved Divergence)", kernel_b_interleaved),
                    ("C (Warp-Aligned)", kernel_c_warp_aligned)]:
        results[name] = bench(k, d_y, T1_N)

    base = results["A (Uniform)"]
    print(f"{'Kernel':<30}{'Avg Time (ms)':>15}{'Slowdown vs A':>16}")
    for name, t in results.items():
        print(f"{name:<30}{t:>15.4f}{t / base:>15.2f}x")
    return results


# ===========================================================================
# Task 2: 1D stencil with halo replication
# ===========================================================================
@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)
    if idx < N:
        if idx == 0:
            left = d_in[0]
        else:
            left = d_in[idx - 1]
        if idx == N - 1:
            right = d_in[N - 1]
        else:
            right = d_in[idx + 1]
        d_out[idx] = 0.25 * left + 0.5 * d_in[idx] + 0.25 * right


def run_stencil(h_in):
    h_in = np.ascontiguousarray(h_in, dtype=np.float32)
    N = h_in.shape[0]
    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array(N, dtype=np.float32)
    threads = 256
    blocks = (N + threads - 1) // threads
    stencil_1d[blocks, threads](d_in, d_out, N)
    cuda.synchronize()
    return d_out.copy_to_host()


def cpu_stencil(arr):
    padded = np.pad(arr, (1, 1), mode='edge')
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


def run_task2():
    N = 100_007
    h_in = np.random.rand(N).astype(np.float32)
    h_out_gpu = run_stencil(h_in)
    cpu_ref = cpu_stencil(h_in)
    assert np.allclose(h_out_gpu, cpu_ref, atol=1e-4)
    delta = float(np.max(np.abs(h_out_gpu - cpu_ref)))
    print(f"TASK 2 PASSED: MAX DELTA = {delta}")


# ===========================================================================
# Task 3: grid-stride vector scaling
# ===========================================================================
@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):
    h_arr = np.ascontiguousarray(h_arr, dtype=np.float32)
    N = h_arr.shape[0]
    d_arr = cuda.to_device(h_arr)
    threads_per_block = 256
    blocks_per_grid = 64  # 16,384 total threads, independent of N
    grid_stride_scale_kernel[blocks_per_grid, threads_per_block](
        d_arr, np.float32(factor), N)
    cuda.synchronize()
    return d_arr.copy_to_host()


def run_task3():
    N = 2 ** 24  # 16,777,216
    factor = 3.5
    res = run_grid_stride(np.ones(N, dtype=np.float32), factor)
    assert res.shape[0] == N
    assert np.all(res == np.float32(factor)), "Not all elements match factor"
    print(f"TASK 3 PASSED: all {N:,} elements == {factor} "
          f"(launched {256 * 64:,} threads)")


# ===========================================================================
# Task 4: 2D Sobel-X
# ===========================================================================
@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)
    if row < rows and col < cols:
        if 0 < row < rows - 1 and 0 < col < cols - 1:
            d_out[row, col] = (
                -1.0 * d_in[row - 1, col - 1] + 1.0 * d_in[row - 1, col + 1]
                - 2.0 * d_in[row, col - 1] + 2.0 * d_in[row, col + 1]
                - 1.0 * d_in[row + 1, col - 1] + 1.0 * d_in[row + 1, col + 1]
            )
        else:
            d_out[row, col] = 0.0


def run_sobel(h_img):
    h_img = np.ascontiguousarray(h_img, dtype=np.float32)
    rows, cols = h_img.shape
    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array((rows, cols), dtype=np.float32)
    threads_2d = (16, 16)
    blocks_2d = (math.ceil(cols / threads_2d[0]),
                 math.ceil(rows / threads_2d[1]))
    sobel_x_kernel[blocks_2d, threads_2d](d_in, d_out, rows, cols)
    cuda.synchronize()
    return d_out.copy_to_host()


def run_task4():
    img = np.random.rand(2048, 2048).astype(np.float32)
    out = run_sobel(img)
    ref = np.zeros_like(img)
    ref[1:-1, 1:-1] = (
        -img[:-2, :-2] + img[:-2, 2:]
        - 2 * img[1:-1, :-2] + 2 * img[1:-1, 2:]
        - img[2:, :-2] + img[2:, 2:]
    )
    assert np.allclose(out, ref, atol=1e-4)
    print("TASK 4 PASSED: MAX DELTA =", float(np.max(np.abs(out - ref))))


# ===========================================================================
# Verification + integrity token (same checks as verify_submission.py)
# ===========================================================================
def verify():
    print("=" * 60)
    print("RUNNING CUDA LAB 02 AUTONOMOUS VERIFICATION")
    print("=" * 60)
    student_id = input("Enter your Student ID: ").strip()
    if not student_id:
        print("[FAIL] Student ID cannot be empty.")
        sys.exit(1)

    try:
        N = 10007
        test_in = np.sin(np.linspace(0, 10, N)).astype(np.float32)
        h_gpu = run_stencil(test_in)
        h_cpu = cpu_stencil(test_in)
        assert np.allclose(h_gpu, h_cpu, atol=1e-4), "Task 2 output mismatch against CPU"
        print("[PASS] Task 2 (1D Stencil & Clamping)")
    except Exception as e:
        print(f"[FAIL] Task 2: {e}")
        sys.exit(1)

    try:
        res = run_grid_stride(np.ones(100000, dtype=np.float32), 4.25)
        assert np.allclose(res, 4.25), "Task 3 elements not uniformly scaled"
        print("[PASS] Task 3 (Grid-Stride Scaling)")
    except Exception as e:
        print(f"[FAIL] Task 3: {e}")
        sys.exit(1)

    try:
        sobel_res = run_sobel(np.ones((64, 64), dtype=np.float32))
        assert np.max(np.abs(sobel_res[1:-1, 1:-1])) < 1e-5, "Task 4 interior gradient != 0 for flat field"
        assert np.all(sobel_res[0, :] == 0.0), "Task 4 border rows not zeroed"
        assert np.all(sobel_res[:, 0] == 0.0), "Task 4 border columns not zeroed"
        print("[PASS] Task 4 (2D Sobel Horizontal)")
    except Exception as e:
        print(f"[FAIL] Task 4: {e}")
        sys.exit(1)

    hasher = hashlib.sha256()
    hasher.update(student_id.encode('utf-8'))
    try:
        device_name = cuda.get_current_device().name
        hasher.update(device_name if isinstance(device_name, bytes) else device_name.encode('utf-8'))
    except Exception:
        hasher.update(b"UNKNOWN_CUDA_DEVICE")
    hasher.update(h_gpu[:32].tobytes())
    hasher.update(sobel_res[:8, :8].tobytes())
    token = hasher.hexdigest()[:20].upper()

    print("\n" + "=" * 60)
    print("VERIFICATION SUCCESSFUL")
    print(f"OFFICIAL SUBMISSION TOKEN: {token}")
    print("=" * 60)
    print("Copy this token directly into your README.md.\n")


# ===========================================================================
if __name__ == "__main__":
    try:
        dev = cuda.get_current_device()
        print("GPU:", dev.name if isinstance(dev.name, str) else dev.name.decode(),
              "| Compute Capability:", dev.compute_capability)
    except Exception as e:
        print("No CUDA device found:", e)
        sys.exit(1)

    print("\n--- Task 1: Warp Divergence ---")
    run_task1()
    print("\n--- Task 2: 1D Stencil ---")
    run_task2()
    print("\n--- Task 3: Grid-Stride ---")
    run_task3()
    print("\n--- Task 4: Sobel-X ---")
    run_task4()
    print("\n--- Verification ---")
    verify()
