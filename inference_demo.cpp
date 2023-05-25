#include <kernel.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>


float check_value(
  const std::vector<__half> & pre_value,
  const std::vector<float> & true_value
) {
  // compare value of pre_value and true_value
  float max_diff = 0.0f;
  for (int i = 0; i < pre_value.size(); ++i) {
    float diff = fabs(__half2float(pre_value[i]) - true_value[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  if (max_diff > 0.5) {
    std::cout << RED 
      << "max diff: " << max_diff << ", check result is failed!"
      << NONE 
      << std::endl;
  } else {
    std::cout << GREEN 
      << "max diff: " << max_diff << ", check result is passed!"
      << NONE 
      << std::endl;
  }
  return max_diff;
}


void compare_value(
  Kernel &kernel,
  const std::vector<std::vector<__half>> &h_output,
  std::string output_dir
  ) {
  float max_diff = 0.0f;
  for (int i = 0; i < 28; ++i) {
    auto present_key_name = kernel.tensor_names_[i * 2 + kernel.n_input_];
    auto present_value_name = kernel.tensor_names_[i * 2 + kernel.n_input_ + 1];
    xt::xarray<float> x_present_key = xt::load_npy<float>(
      output_dir + std::string(present_key_name) + std::string(".npy")
    );
    xt::xarray<float> x_present_value = xt::load_npy<float>(
      output_dir + std::string(present_value_name) + std::string(".npy")
    );
    std::vector<float> h_present_key(x_present_key.begin(), x_present_key.end());
    std::vector<float> h_present_value(x_present_value.begin(), x_present_value.end());
    std::cout << "present key name: " << present_key_name << std::endl;
    float diff1 = check_value(h_output[i * 2], h_present_key);
    std::cout << "present value name: " << present_value_name << std::endl;
    float diff2 = check_value(h_output[i * 2 + 1], h_present_value);
    if (diff1 > max_diff) {
      max_diff = diff1;
    }
    if (diff2 > max_diff) {
      max_diff = diff2;
    }
    std::cout << std::endl;
  }
  // compare logists
  xt::xarray<float> x_logits = xt::load_npy<float>(
    output_dir + "logits.npy"
  );
  std::vector<float> h_logits(x_logits.begin(), x_logits.end());
  std::cout << "logits name: " << kernel.tensor_names_[kernel.n_input_ + 56] << std::endl;
  float diff3 = check_value(h_output[56], h_logits);
  if (diff3 > max_diff) {
    max_diff = diff3;
  }
  std::cout << "max diff: " << max_diff << std::endl;
}


int test1(Kernel &kernel) {
  // test data for context1
  // load input data
  xt::xarray<int> input_ids = xt::load_npy<int>(
    "output/np_input1/input_ids.npy"
  );
  xt::xarray<int> position_ids = xt::load_npy<int>(
    "output/np_input1/position_ids.npy"
  );
  xt::xarray<bool> attention_mask = xt::load_npy<bool>(
    "output/np_input1/attention_mask.npy"
  );
  std::vector<int> h_input_ids(input_ids.begin(), input_ids.end());
  std::vector<int> h_position_ids(position_ids.begin(), position_ids.end());
  std::vector<bool> h_attention_mask(attention_mask.begin(), attention_mask.end());
  std::vector<std::vector<__half>> h_output = kernel.forward(
    h_input_ids,
    h_position_ids,
    h_attention_mask
  );
  compare_value(kernel, h_output, "output/np_output1/");
}


int test2(Kernel &kernel) {
  std::vector<int> h_input_ids({19316});
  std::vector<int> h_position_ids({2, 2});
  std::vector<bool> h_attention_mask = {false};
  std::vector<std::vector<std::vector<__half>>> past_key_values;
  int past_seq_length = 6;
  for (int i = 0; i < 28; ++i) {
    std::vector<std::vector<__half>> tmp;
    for (int j = 0; j < 2; ++j) {
      std::vector<__half> tmp2(past_seq_length * 1 * 32 * 128);
      tmp.push_back(tmp2);
    }
    past_key_values.push_back(tmp);
  }
  std::vector<std::vector<__half>> result_list = kernel.forward(
    h_input_ids,
    h_position_ids,
    h_attention_mask,
    past_key_values
  );
}


int main() {
  Kernel kernel("models/chatglm6b-bs1-12.5G.plan", 1);
  test1(kernel);
  // test2(kernel);
}