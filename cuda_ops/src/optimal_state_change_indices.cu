#include "common.hpp"

__global__ void SingleStateChangeIndicesKernel(
    const float* state_tensor,
    const float* action_tensor,
    const int* lens,
    int* out_indices,
    const int max_action_state_distance
) {
    const int batch_idx = blockIdx.x;
    const int video_len = blockDim.x;
    const int state1_pos = threadIdx.x;
    const int actual_len = lens[batch_idx];

    // get pointer to shared memory
    extern __shared__ char shared_mem[];
    int* state1_to_action_pos = reinterpret_cast<int*>(shared_mem);
    int* state1_to_state2_pos = state1_to_action_pos + video_len;
    float* state1_to_score = reinterpret_cast<float*>(state1_to_state2_pos + video_len);
    float* action_tensor_shared = state1_to_score + video_len;
    float* state_tensor_shared = action_tensor_shared + video_len;

    // load action and state tensors into shared memory
    action_tensor_shared[state1_pos] = action_tensor[batch_idx * video_len + state1_pos];
    state_tensor_shared[2 * state1_pos + 0] = state_tensor[batch_idx * video_len * 2 + state1_pos * 2 + 0];
    state_tensor_shared[2 * state1_pos + 1] = state_tensor[batch_idx * video_len * 2 + state1_pos * 2 + 1];

    __syncthreads();

    float best_score = -std::numeric_limits<float>::infinity();
    int best_action_pos = 0, best_state2_pos = 0; // position of states/action for videos shorter than 3

    for (int action_pos = state1_pos + 1; action_pos <= state1_pos + max_action_state_distance && action_pos < actual_len - 1; ++action_pos) { // -1: need at least one position for state2
        float action_score = action_tensor_shared[action_pos];

        for (int state2_pos = action_pos + 1; state2_pos <= action_pos + max_action_state_distance && state2_pos < actual_len; ++state2_pos) {
            float state2_score = state_tensor_shared[2 * state2_pos + 1]; // 2 states, +1 for second state

            float score = action_score * state2_score;
            if (score > best_score) {
                best_score = score;
                best_action_pos = action_pos;
                best_state2_pos = state2_pos;
            }
        }
    }

    state1_to_action_pos[state1_pos] = best_action_pos;
    state1_to_state2_pos[state1_pos] = best_state2_pos;
    state1_to_score[state1_pos] = best_score * state_tensor_shared[2 * state1_pos + 0];

    __syncthreads();

    if (state1_pos == 0) { // compute reduction only on the first thread
        best_score = state1_to_score[0];
        int best_state1_pos = 0;
        for (int i = 1; i < actual_len - 2; ++i) { // -2: need at least one position for action and one for state2
            if (best_score < state1_to_score[i]) {
                best_state1_pos = i;
                best_score = state1_to_score[i];
            }
        }
        best_action_pos = state1_to_action_pos[best_state1_pos];
        best_state2_pos = state1_to_state2_pos[best_state1_pos];

        out_indices[batch_idx * 3 + 0] = best_state1_pos;
        out_indices[batch_idx * 3 + 1] = best_state2_pos;
        out_indices[batch_idx * 3 + 2] = best_action_pos;
    }
}

torch::Tensor optimal_state_change_indices(
    torch::Tensor state_tensor, torch::Tensor action_tensor, torch::Tensor lens, int max_action_state_distance) {

    CHECK_CUDA_INPUT(state_tensor);
    CHECK_CUDA_INPUT(action_tensor);
    CHECK_CUDA_INPUT(lens);

    int batch_size = state_tensor.size(0);
    int video_len = state_tensor.size(1);

    TORCH_CHECK(state_tensor.size(2) == 2, "state_tensor must be of shape [batch, video_len, 2]")
    TORCH_CHECK(action_tensor.size(2) == 1, "action_tensor must be of shape [batch, video_len, 1]")

    auto options = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    auto out_indices = torch::zeros({batch_size, 3}, options);

    const int threads = video_len;
    const int blocks = batch_size;
    // store in shared memory:
    //  best action position for each state1 position (1x int)
    //  best state2 position for each state1 position (1x int)
    //  best score for each state1 position (1x float)
    //  action tensor (1x float)
    //  state tensor (2x float)
    const int shared_mem = video_len * (2 * sizeof(int) + 4 * sizeof(float));
    SingleStateChangeIndicesKernel<<<blocks, threads, shared_mem>>>(
            state_tensor.data_ptr<float>(),
            action_tensor.data_ptr<float>(),
            lens.data_ptr<int>(),
            out_indices.data_ptr<int>(),
            max_action_state_distance);

    return out_indices;
}
