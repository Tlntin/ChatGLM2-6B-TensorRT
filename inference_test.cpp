#include <kernel.hpp>

float check_value(
  const torch::Tensor & pre_value,
  const torch::Tensor & true_value
) {
  if (pre_value.sizes() != true_value.sizes()) {
    std::cout << RED 
      << "pre_value.size() != true_value.size(), check result is failed!"
      << NONE 
      << std::endl;
    throw std::runtime_error("pre_value.size() != true_value.size()");
  }
  // compare value of pre_value and true_value
  float max_diff = (pre_value - true_value).abs_().max().item<float>();
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
  const std::vector<torch::Tensor> &d_output,
  std::string file_path
  ) {
  if (!std::filesystem::exists(file_path)) {
    std::cerr << RED << file_path << " not exsits!" << NONE << std::endl;
    throw std::runtime_error("file not exesits");
  }
  torch::jit::script::Module container = torch::jit::load(file_path);
  torch::Device device = d_output[0].device();
  float max_diff = 0.0f;
  for (int i = 0; i < 28; ++i) {
    auto present_key_name = kernel.tensor_names_[i * 2 + kernel.n_input_];
    auto present_value_name = kernel.tensor_names_[i * 2 + kernel.n_input_ + 1];
    torch::Tensor present_key = container.attr(present_key_name).toTensor().to(device);
    torch::Tensor present_value = container.attr(present_value_name).toTensor().to(device);
    std::cout << "present key name: " << present_key_name << std::endl;
    float diff1 = check_value(d_output[i * 2], present_key);
    std::cout << "present value name: " << present_value_name << std::endl;
    float diff2 = check_value(d_output[i * 2 + 1], present_value);
    if (diff1 > max_diff) {
      max_diff = diff1;
    }
    if (diff2 > max_diff) {
      max_diff = diff2;
    }
    std::cout << std::endl;
  }
  // compare logists
  std::cout << "check logitst diff" << std::endl;
  torch::Tensor logits = container.attr("logits").toTensor().to(device);
  float diff3 = check_value(d_output[kernel.n_output_ - 1], logits);
  if (diff3 > max_diff) {
    max_diff = diff3;
  }
  std::cout << "max diff: " << max_diff << std::endl;
}


int test1(Kernel &kernel) {
  // test data for context1
  // load input data
  if (!torch::cuda::is_available()) {
    std::cout << "cuda is not available!" << std::endl;
    throw std::runtime_error("cuda is not available!");
  }
  torch::Device device(torch::kCUDA);
  std::string input_path = "../output/pt_input1.pt";
  if (!std::filesystem::exists(input_path)) {
    std::cerr << RED << "input path " << input_path << "not exists" << NONE << std::endl;
    throw std::runtime_error("input path not exsist!");
  }
  torch::jit::script::Module container = torch::jit::load(input_path);
  torch::Tensor input_ids = container.attr("input_ids").toTensor().to(device);
  torch::Tensor position_ids = container.attr("position_ids").toTensor().to(device);
  torch::Tensor attention_mask = container.attr("attention_mask").toTensor().to(device);
  std::vector<torch::Tensor> input_tensors({input_ids, position_ids, attention_mask});
  std::vector<torch::Tensor> d_output = kernel.forward(input_tensors);
  std::cout << "output size: " << d_output.size() << std::endl;
  std::cout << "==================================" << std::endl;
  std::cout << "start compare value for test1" << std::endl;
  compare_value(kernel, d_output, "../output/pt_output1.pt");
  std::cout << "compare value for test1 done!" << std::endl;
  std::cout << "==================================" << std::endl;
  std::cout << std::endl;
  return 0;
}

int test2(Kernel &kernel) {
  // test data for context1
  // load input data
  if (!torch::cuda::is_available()) {
    std::cout << "cuda is not available!" << std::endl;
    throw std::runtime_error("cuda is not available!");
  }
  torch::Device device(torch::kCUDA);
  std::string input_path = "../output/pt_input2.pt";
  if (!std::filesystem::exists(input_path)) {
    std::cerr << RED << "input path " << input_path << "not exists" << NONE << std::endl;
    throw std::runtime_error("input path not exsist!");
  }
  torch::jit::script::Module container = torch::jit::load(input_path);
  torch::Tensor input_ids = container.attr("input_ids").toTensor().to(device);
  torch::Tensor position_ids = container.attr("position_ids").toTensor().to(device);
  torch::Tensor attention_mask = container.attr("attention_mask").toTensor().to(device);
  std::cout << "loading past key and value" << std::endl;
  std::vector<torch::Tensor> input_tensors({input_ids, position_ids, attention_mask});
  // load past key and value
  for (int i = 0; i < 28; ++i) {
    std::string past_key_name = \
      "past_key_values."+ std::to_string(i) + ".decorder.key";
    std::string past_value_name = \
      "past_key_values." + std::to_string(i) + ".decorder.value";
    torch::Tensor past_key = container.attr(past_key_name).toTensor().to(device);
    torch::Tensor past_value = container.attr(past_value_name).toTensor().to(device);
    input_tensors.push_back(past_key);
    input_tensors.push_back(past_value);
  }
  // for (int i = 0; i < kernel.n_input_; ++i) {
  //   std::cout << "tensor name " << i << ": " << kernel.tensor_names_[i] << std::endl;
  //   std::cout << "input tensor " << i << " size: " << input_tensors[i].sizes() << std::endl;
  // }
  std::cout << "load past key and value done!" << std::endl;
  std::vector<torch::Tensor> d_output = kernel.forward(input_tensors);
  std::cout << "==================================" << std::endl;
  std::cout << "start compare value for test2" << std::endl;
  compare_value(kernel, d_output, "../output/pt_output2.pt");
  std::cout << "compare value for test2 done!" << std::endl;
  std::cout << "==================================" << std::endl;
  return 0;
}

int main() {
  Kernel kernel("../models/chatglm6b-bs1-12.5G.plan", 1);
  test1(kernel);
  test2(kernel);
}