#include <kernel.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>


float check_value(
  const std::vector<__half> & pre_value,
  const std::vector<float> & true_value
) {
  if (pre_value.size() != true_value.size()) {
    std::cout << RED 
      << "pre_value.size() != true_value.size(), check result is failed!"
      << NONE 
      << std::endl;
    throw std::runtime_error("pre_value.size() != true_value.size()");
  }
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
  std::cout << "==================================" << std::endl;
  std::cout << "start compare value for test1" << std::endl;
  compare_value(kernel, h_output, "output/np_output1/");
  std::cout << "compare value for test1 done!" << std::endl;
  std::cout << "==================================" << std::endl;
  std::cout << std::endl;
  return 0;
}


int test2(Kernel &kernel) {
  // test data for context1
  // load input data
  xt::xarray<int> input_ids = xt::load_npy<int>(
    "output/np_input2/input_ids.npy"
  );
  xt::xarray<int> position_ids = xt::load_npy<int>(
    "output/np_input2/position_ids.npy"
  );
  xt::xarray<bool> attention_mask = xt::load_npy<bool>(
    "output/np_input2/attention_mask.npy"
  );
  std::vector<int> h_input_ids(input_ids.begin(), input_ids.end());
  std::vector<int> h_position_ids(position_ids.begin(), position_ids.end());
  std::vector<bool> h_attention_mask(attention_mask.begin(), attention_mask.end());
  std::vector<std::vector<std::vector<__half>>> past_key_values;
  // load past key and value
  std::cout << "loading past key and value" << std::endl;
  for (int i = 0; i < 28; ++i) {
    std::string past_key_path = \
      "output/np_input2/past_key_values."+ std::to_string(i) + ".decorder.key.npy";
    std::string past_value_path = \
      "output/np_input2/past_key_values." + std::to_string(i) + ".decorder.value.npy";
    xt::xarray<float> x_past_key = xt::load_npy<float>(past_key_path);
    xt::xarray<float> x_past_value = xt::load_npy<float>(past_value_path);
    std::vector<float> h_past_key(x_past_key.begin(), x_past_key.end());
    std::vector<float> h_past_value(x_past_value.begin(), x_past_value.end());
    std::vector<__half> h_past_key_half(h_past_key.size());
    std::vector<__half> h_past_value_half(h_past_value.size());
    for (int i = 0; i < h_past_key.size(); i++) {
      h_past_key_half[i] = __float2half(h_past_key[i]);
    }
    for (int i = 0; i < h_past_value.size(); i++) {
      h_past_value_half[i] = __float2half(h_past_value[i]);
    }
    std::vector<std::vector<__half>> temp_({h_past_key_half, h_past_value_half});
    past_key_values.push_back(temp_);
  }
  std::cout << "load past key and value done!" << std::endl;
  std::vector<std::vector<__half>> h_output = kernel.forward(
    h_input_ids,
    h_position_ids,
    h_attention_mask,
    past_key_values
  );
  std::cout << "==================================" << std::endl;
  std::cout << "start compare value for test2" << std::endl;
  compare_value(kernel, h_output, "output/np_output2/");
  std::cout << "compare value for test2 done!" << std::endl;
  std::cout << "==================================" << std::endl;
  return 0;
}


int main() {
  Kernel kernel("models/chatglm6b-bs1-12.5G.plan", 1);
  test1(kernel);
  test2(kernel);
}