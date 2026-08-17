import torch
import time
import os
import sys

# Add parent directory to path to ensure proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ascii_engine import ASCIIEngine, HAS_CUDA_EXTENSION, tile_voting_ext

def benchmark_tile_voting():
    print("--- STARTING TILED VOTING BENCHMARK ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # 1. Setup mock inputs
    # A typical video frame resolution: 1080p (1080 x 1920)
    h, w = 1080, 1920
    edge_threshold = 12
    
    # Create random directions tensor containing values from -1 to 3
    directions = torch.randint(-1, 4, (h, w), device=device, dtype=torch.int64)
    
    # 2. Run pure PyTorch implementation (fallback)
    def pytorch_tile_voting(dirs, threshold):
        tiles = dirs.view(h // 8, 8, w // 8, 8).permute(0, 2, 1, 3).contiguous().view(h // 8, w // 8, 64)
        counts = torch.stack([
            (tiles == 0).sum(dim=-1),
            (tiles == 1).sum(dim=-1),
            (tiles == 2).sum(dim=-1),
            (tiles == 3).sum(dim=-1)
        ], dim=-1)
        max_counts, dominant = torch.max(counts, dim=-1)
        dominant[max_counts < threshold] = -1
        return dominant

    if not HAS_CUDA_EXTENSION:
        print("\nNote: Custom CUDA/C++ extension is not loaded (requires Visual Studio C++ Build Tools).")
        print("Benchmarking the active pure PyTorch fallback implementation instead:")
        
        print("Running warm-up passes...")
        for _ in range(50):
            _ = pytorch_tile_voting(directions, edge_threshold)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        print("Benchmarking pure PyTorch implementation...")
        t_start = time.perf_counter()
        for _ in range(1000):
            _ = pytorch_tile_voting(directions, edge_threshold)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_pytorch = (time.perf_counter() - t_start) / 1000.0
        print(f"PyTorch Fallback Average Time (1080p frame): {t_pytorch * 1000:.3f} ms")
        print("---------------------------------------")
        return


    # 3. Verify correctness
    print("\nVerifying numerical equivalence...")
    out_pytorch = pytorch_tile_voting(directions, edge_threshold)
    out_extension = tile_voting_ext.tile_voting(directions, edge_threshold)
    
    if torch.equal(out_pytorch, out_extension):
        print("Success: CUDA/C++ extension output matches pure PyTorch output perfectly!")
    else:
        print("ERROR: Numerical outputs mismatch!")
        print(f"PyTorch output: \n{out_pytorch[:5, :5]}")
        print(f"Extension output: \n{out_extension[:5, :5]}")
        return

    # 4. Benchmark performance
    print("\nRunning warm-up passes...")
    for _ in range(50):
        _ = pytorch_tile_voting(directions, edge_threshold)
        _ = tile_voting_ext.tile_voting(directions, edge_threshold)
    torch.cuda.synchronize()

    print("Benchmarking pure PyTorch implementation...")
    t_start = time.perf_counter()
    for _ in range(1000):
        _ = pytorch_tile_voting(directions, edge_threshold)
    torch.cuda.synchronize()
    t_pytorch = (time.perf_counter() - t_start) / 1000.0
    print(f"PyTorch Average Time: {t_pytorch * 1000:.3f} ms")

    print("Benchmarking Custom CUDA/C++ extension...")
    t_start = time.perf_counter()
    for _ in range(1000):
        _ = tile_voting_ext.tile_voting(directions, edge_threshold)
    torch.cuda.synchronize()
    t_ext = (time.perf_counter() - t_start) / 1000.0
    print(f"Extension Average Time: {t_ext * 1000:.3f} ms")

    speedup = t_pytorch / t_ext
    print(f"\nSpeedup: {speedup:.2f}x faster!")
    print("---------------------------------------")

if __name__ == "__main__":
    benchmark_tile_voting()
