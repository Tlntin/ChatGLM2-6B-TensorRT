#include <kernel.hpp>

int test1(Kernel &kernel) {
  // test data for context1
  std::vector<int> h_input_ids({1, 5, 74874, 130001});
  std::vector<int> h_position_ids({
    0, 1, 2, 2, 
    0, 0, 0, 1
  });
  std::vector<bool> h_attention_mask({
    false, false, false, true, 
    false, false, false, true, 
    false, false, false, true,
    false, false, false, false
  });
  std::vector<std::vector<__half>> h_output = kernel.forward(
    h_input_ids,
    h_position_ids,
    h_attention_mask
  );
  return 0;
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
  // test1();
  Kernel kernel("models/chatglm6b-bs1-12.5G.plan", 1);
  test1(kernel);
  test2(kernel);
}