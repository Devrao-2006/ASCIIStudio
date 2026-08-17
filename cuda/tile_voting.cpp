#include <torch/extension.h>

// Declaration of CUDA launcher
torch::Tensor tile_voting_cuda(torch::Tensor directions, int edge_threshold);

// Optimized CPU implementation
torch::Tensor tile_voting_cpu(torch::Tensor directions, int edge_threshold) {
    const int height = directions.size(0);
    const int width = directions.size(1);
    const int grid_h = height / 8;
    const int grid_w = width / 8;
    
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(directions.device());
    auto dominant_dirs = torch::empty({grid_h, grid_w}, options);
    
    const int64_t* dir_ptr = directions.data_ptr<int64_t>();
    int64_t* dom_ptr = dominant_dirs.data_ptr<int64_t>();
    
    // Process tiles sequentially on CPU
    for (int ty = 0; ty < grid_h; ++ty) {
        for (int tx = 0; tx < grid_w; ++tx) {
            int counts[4] = {0, 0, 0, 0};
            for (int dy = 0; dy < 8; ++dy) {
                for (int dx = 0; dx < 8; ++dx) {
                    int gy = ty * 8 + dy;
                    int gx = tx * 8 + dx;
                    int64_t val = dir_ptr[gy * width + gx];
                    if (val >= 0 && val <= 3) {
                        counts[val]++;
                    }
                }
            }
            int max_val = -1;
            int dominant_dir = -1;
            for (int i = 0; i < 4; ++i) {
                if (counts[i] > max_val) {
                    max_val = counts[i];
                    dominant_dir = i;
                }
            }
            if (max_val < edge_threshold) {
                dominant_dir = -1;
            }
            dom_ptr[ty * grid_w + tx] = dominant_dir;
        }
    }
    return dominant_dirs;
}

// Wrapper to route call depending on input tensor device type
torch::Tensor tile_voting(torch::Tensor directions, int edge_threshold) {
    if (directions.is_cuda()) {
        return tile_voting_cuda(directions, edge_threshold);
    } else {
        return tile_voting_cpu(directions, edge_threshold);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tile_voting", &tile_voting, "8x8 Tile Voting Step (CPU/CUDA)");
}
