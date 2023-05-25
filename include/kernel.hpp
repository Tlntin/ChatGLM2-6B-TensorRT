#include <iostream>
#include <tuple>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <memory>
#include <color.h>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <cuda_fp16.h>


class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char* msg) noexcept override;
};


std::tuple<const char *, std::size_t> get_data_type(int data_type);


#define CHECK_CUDA(call) _check_cuda(call, #call, __FILE__, __LINE__)
void _check_cuda(cudaError_t code, const char *func, const char *file, int line);


bool load_engine_data(const std::string engine_path, std::vector<char> & engine_data);


class Kernel {
  private:
    Logger logger_;
    std::shared_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_1_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_2_;
    cudaStream_t stream_1_;
    cudaStream_t stream_2_;
  public:
    const int batch_size_;
    const std::size_t n_total_ = 116;
    const std::size_t n_input_ = 59;
    const std::size_t n_output_ = 57;
    std::vector<char const *> tensor_names_;
    Kernel(const std::string engine_path, int batch_size);
    ~Kernel();
    void load_engine(const std::string engine_path);
    void vertify_io_number();
    void init_execute_context();
    std::vector<std::vector<__half>> forward(
      std::vector<int> & input_ids,
      std::vector<int> & position_ids, 
      std::vector<bool> & attention_mask
    );
    std::vector<std::vector<__half>> forward(
      std::vector<int> & input_ids,
      std::vector<int> & position_ids, 
      std::vector<bool> & attention_mask,
      std::vector<std::vector<std::vector<__half>>> & past_key_values
    );
    void set_input_for_context1(const int seq_length);
    void set_input_for_context2(const int past_seq_length);
    void get_tensor_size(
      std::shared_ptr<nvinfer1::IExecutionContext> & context,
      std::vector<std::size_t> & bytes_list,
      std::vector<std::size_t> & type_bytes_list
    );
    std::vector<std::vector<__half>> run_gpu_inference(
      const std::vector<int> & input_ids,
      const std::vector<int> & position_ids,
      const std::vector<char> & attention_mask,
      const std::vector<std::vector<std::vector<__half>>> & past_key_values, 
      std::vector<std::size_t> & bytes_list,
      std::vector<std::size_t> & type_bytes_list,
      std::shared_ptr<nvinfer1::IExecutionContext> context, 
      cudaStream_t & stream
    );
};