#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void tile_voting_cuda_kernel(
    const int64_t* __restrict__ directions,
    int64_t* __restrict__ dominant_dirs,
    int width, int height, int edge_threshold)
{
    // Shared memory to accumulate votes for each direction (0, 1, 2, 3) in the 8x8 tile
    __shared__ int counts[4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Thread (0,0) clears the accumulator for this block
    if (tx == 0 && ty == 0) {
        counts[0] = 0;
        counts[1] = 0;
        counts[2] = 0;
        counts[3] = 0;
    }
    __syncthreads();

    // Map threads to global coordinates
    int global_x = blockIdx.x * 8 + tx;
    int global_y = blockIdx.y * 8 + ty;

    if (global_x < width && global_y < height) {
        int64_t val = directions[global_y * width + global_x];
        // Only count valid direction indices (0, 1, 2, 3)
        if (val >= 0 && val <= 3) {
            atomicAdd(&counts[val], 1);
        }
    }
    __syncthreads();

    // Thread (0,0) finds the winning direction and writes the result
    if (tx == 0 && ty == 0) {
        int max_val = -1;
        int dominant_dir = -1;
        
        for (int i = 0; i < 4; ++i) {
            if (counts[i] > max_val) {
                max_val = counts[i];
                dominant_dir = i;
            }
        }
        
        // If the vote count is below the threshold, it is not considered an edge
        if (max_val < edge_threshold) {
            dominant_dir = -1;
        }
        
        int tile_y = blockIdx.y;
        int tile_x = blockIdx.x;
        int grid_width = (width + 7) / 8;
        dominant_dirs[tile_y * grid_width + tile_x] = dominant_dir;
    }
}

torch::Tensor tile_voting_cuda(
    torch::Tensor directions,
    int edge_threshold) 
{
    const int height = directions.size(0);
    const int width = directions.size(1);
    
    const int grid_h = (height + 7) / 8;
    const int grid_w = (width + 7) / 8;
    
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(directions.device());
    auto dominant_dirs = torch::empty({grid_h, grid_w}, options);
    
    dim3 threads(8, 8);
    dim3 grid(grid_w, grid_h);
    
    tile_voting_cuda_kernel<<<grid, threads>>>(
        directions.data_ptr<int64_t>(),
        dominant_dirs.data_ptr<int64_t>(),
        width, height, edge_threshold
    );
    
    return dominant_dirs;
}
