#include <iostream>
#include <tuple>
#include <fstream>
#include <NvInfer.h>
#include <vector>
#include <cuda_runtime.h>
#include <memory>
#include <color.h>
#include <cuda_fp16.h>


class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char* msg) noexcept override
  {
    // suppress info-level messages
    switch (severity)
    {
    case Severity::kINTERNAL_ERROR: {
      std::cerr << RED << "INTERNAL_ERROR: " << msg << NONE << std::endl; break;
      break;
    }
    case Severity::kERROR: {
      std::cerr << RED << "ERROR: " << msg << NONE << std::endl; break;
      break;
    } 
    case Severity::kWARNING: {
      std::cerr << YELLOW << "WARNING: " << msg << NONE << std::endl; break;
      break;
    }
    case Severity::kINFO: {
      std::cerr << GREEN << "INFO: " << msg << NONE << std::endl; break;
      break;
    }
    default:
      break;
    }
  }
};


std::tuple<const char *, std::size_t> get_data_type(int data_type) {
  switch (data_type) {
    case 0: return std::make_tuple("float32", 4);
    //! 32-bit floating point format.
    // kFLOAT = 0,

    case 1: return std::make_tuple("float16", 2);
    //! IEEE 16-bit floating-point format.
    // kHALF = 1,

    case 2: return std::make_tuple("int8", 1);
    //! Signed 8-bit integer representing a quantized floating-point value.
    // kINT8 = 2,

    case 3: return std::make_tuple("int32", 4);
    //! Signed 32-bit integer format.
    // kINT32 = 3,

    case 4: return std::make_tuple("bool", sizeof(bool));
    //! 8-bit boolean. 0 = false, 1 = true, other values undefined.
    // kBOOL = 4,

    case 5: return std::make_tuple("uint8", 1);
    //! Unsigned 8-bit integer format.
    //! Cannot be used to represent quantized floating-point values.
    //! Use the IdentityLayer to convert kUINT8 network-level inputs to {kFLOAT, kHALF} prior
    //! to use with other TensorRT layers, or to convert intermediate output
    //! before kUINT8 network-level outputs from {kFLOAT, kHALF} to kUINT8.
    //! kUINT8 conversions are only supported for {kFLOAT, kHALF}.
    //! kUINT8 to {kFLOAT, kHALF} conversion will convert the integer values
    //! to equivalent floating point values.
    //! {kFLOAT, kHALF} to kUINT8 conversion will convert the floating point values
    //! to integer values by truncating towards zero. This conversion has undefined behavior for
    //! floating point values outside the range [0.0f, 256.0f) after truncation.
    //! kUINT8 conversions are not supported for {kINT8, kINT32, kBOOL}.
    // kUINT8 = 5,

    case 6: return std::make_tuple("float8", 1);
    //! Signed 8-bit floating point with
    //! 1 sign bit, 4 exponent bits, 3 mantissa bits, and exponent-bias 7.
    //! \warning kFP8 is not supported yet and will result in an error or undefined behavior.
    // kFP8 = 6

    default: return std::make_tuple("unknown", 0);
  }
}


#define CHECK_CUDA(call) { \
  cudaError_t status = call; \
  if (status != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
    return -1; \
  } \
}

bool load_engine(const std::string engine_path, std::vector<char> & engine_data) {
  std::ifstream engine_file(
    engine_path,
    std::ios::binary | std::ios::in
  );
  engine_file.seekg(0, std::ios::end);
  size_t engine_size = engine_file.tellg();
  std::cout << "engine_size: " << (engine_size >> 20) << "M" << std::endl;
  if (engine_size == 0) {
    std::cerr << "Failed to read engine file" << std::endl;
    engine_file.close();
    return false;
  }
  engine_data.resize(engine_size);
  engine_file.seekg(0, std::ios::beg);
  engine_file.read(engine_data.data(), engine_size);
  engine_file.close();
  return true;
}


int main() {
  Logger logger;
  auto runtime = std::shared_ptr<nvinfer1::IRuntime>(
    nvinfer1::createInferRuntime(logger)
  );

  // begin to load engine
  std::string engine_path = "models/chatglm6b-bs1-12.5G.plan";
  std::vector<char> engine_data;
  bool result1 = load_engine(engine_path, engine_data);
  if (!result1) {
    std::cerr << "Failed to load engine" << std::endl;
    return -1;
  }
  auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
    runtime->deserializeCudaEngine(
      engine_data.data(), engine_data.size(), nullptr
    )
  );
  if (engine == nullptr) {
    std::cerr << "Failed to deserialize engine" << std::endl;
    return -1;
  }  

  // === get input and output == //
  std::size_t n_io = engine->getNbIOTensors(); 
  std::vector<char const *> tensor_names(n_io);
  std::size_t n_input = 0;
  std::size_t n_output = 0;
  for (int i = 0; i < n_io; ++i) {
    tensor_names[i] = engine->getIOTensorName(i);
    if (engine->bindingIsInput(i)) {
      ++n_input;
    } else {
      ++n_output;
    }
  }
  std::cout << "number of input: " << n_input << std::endl;
  std::cout << "number of output: " << n_input << std::endl;
  std::cout << "optimizationprofile count: "
    << engine->getNbOptimizationProfiles()
    << std::endl;

  // === create context == //
  // this context is for inference when past_key_values is None
  auto context1 = std::shared_ptr<nvinfer1::IExecutionContext>(
    engine->createExecutionContext()
  );
  context1->setOptimizationProfile(0);
  cudaStream_t stream1;
  CHECK_CUDA(cudaStreamCreate(&stream1));

  // this context is for inference when past_key_values is not None
  auto context2 = std::shared_ptr<nvinfer1::IExecutionContext>(
    engine->createExecutionContext()
  );
  context2->setOptimizationProfile(1);
  cudaStream_t stream2;
  CHECK_CUDA(cudaStreamCreate(&stream2));
  // set batch size for global context
  int batch_size = 1;

  // === set input shape for context1 == //
  std::cout << "=============================" << std::endl;
  std::cout << "set input shape for context1" << std::endl;
  int seq_length = 4;
  context1->setInputShape(
    "input_ids",
    nvinfer1::Dims2{batch_size, seq_length}
  );
  context1->setInputShape(
    "position_ids",
    nvinfer1::Dims3{batch_size, 2, seq_length}
  );
  context1->setInputShape(
    "attention_mask",
    nvinfer1::Dims4{batch_size, 1,  seq_length, seq_length}
  );
  std::vector<std::size_t> size_list1(n_io);
  std::vector<std::size_t> bytes_list1(n_io);
  for (int i = 0; i < n_io; ++i) {
    // std::cout << "tensor name: " << tensor_names[i] << std::endl;
    auto dims = context1->getTensorShape(tensor_names[i]);
    // std::cout << "tensor shape: ";
    // for (int j = 0; j < dims.nbDims; ++j) {
    //   std::cout << dims.d[j] << ", ";
    // }
    // std::cout << std::endl;
    std::size_t size = 1;
    for (int j = 0; j < dims.nbDims; ++j) {
      size *= dims.d[j];
    }
    // std::cout << "tensor size: " << size << std::endl;
    size_list1[i] = size;
    auto size_result = get_data_type(
      (int)engine->getTensorDataType(tensor_names[i])
    );
    const char * data_type = std::get<0>(size_result);
    std::size_t type_size = std::get<1>(size_result);
    bytes_list1[i] = type_size * size;
    // std::cout << "tensor bytes: " << bytes_list1[i] << std::endl;
    // std::cout << "tensor type: " << data_type << std::endl;
    // std::cout << std::endl;
  }
  std::cout << "=============================" << std::endl;

  // prepare input data for context1
  int h_input_ids[] = {1, 5, 74874, 130001};
  int h_position_ids[] = {
    0, 1, 2, 2, 
    0, 0, 0, 1
  };
  bool h_attention_mask[] = {
    false, false, false, true, 
    false, false, false, true, 
    false, false, false, true,
    false, false, false, false
  };
  int *d_input_ids;
  int *d_position_ids;
  bool *d_attention_mask;
  CHECK_CUDA(cudaMalloc((void **)&d_input_ids, bytes_list1[0]));
  CHECK_CUDA(cudaMalloc((void **)&d_position_ids, bytes_list1[1]));
  CHECK_CUDA(cudaMalloc((void **)&d_attention_mask, bytes_list1[2]));
  CHECK_CUDA(
    cudaMemcpyAsync(
      d_input_ids,
      h_input_ids,
      bytes_list1[0],
      cudaMemcpyHostToDevice,
      stream1
    )
  );
  CHECK_CUDA(
    cudaMemcpyAsync(
      d_position_ids,
      h_position_ids,
      bytes_list1[1],
      cudaMemcpyHostToDevice,
      stream1
    )
  );
  CHECK_CUDA(
    cudaMemcpyAsync(
      d_attention_mask,
      h_attention_mask,
      bytes_list1[2],
      cudaMemcpyHostToDevice,
      stream1
    )
  );
  context1->setTensorAddress(
    "input_ids", d_input_ids
  );
  context1->setTensorAddress(
    "position_ids", d_position_ids
  );
  context1->setTensorAddress(
    "attention_mask", d_attention_mask
  );
  void * d_input[59] = {d_input_ids, d_position_ids, d_attention_mask};
  for (int i = 3; i < n_input; ++i) {
    CHECK_CUDA(cudaMalloc((void **)&d_input[i], 2));
    // set input past_key_value data to zero, given that past_key_value is 1 data
    CHECK_CUDA(cudaMemset(d_input[i], 0, 2));
    context1->setTensorAddress(
      tensor_names[i], d_input[i]
    );
  }
  // prepare output data for context
  __half * h_output[59];
  void * d_output[59];
  for (int i = n_input; i < n_io; ++i) {
    h_output[i - n_input] = new __half[size_list1[i]];
    CHECK_CUDA(
      cudaMalloc((void **)&d_output[i - n_input], bytes_list1[i])
    );
    context1->setTensorAddress(
      tensor_names[i], d_output[i - n_input]
    );
  }
  // run inference for context1 on stream1
  context1->enqueueV3(stream1);

  cudaDeviceSynchronize();
  cudaStreamSynchronize(stream1);
  // copy output data from device to host
  for (int i = n_input; i < n_io; ++i) {
    CHECK_CUDA(
      cudaMemcpyAsync(
        h_output[i - n_input],
        d_output[i - n_input],
        bytes_list1[i],
        cudaMemcpyDeviceToHost,
        stream1
    ));
  }

  cudaDeviceSynchronize();
  cudaStreamSynchronize(stream1);
  std::cout << "output[0] for context1: " << __half2float(h_output[0][0]) << std::endl;

  // free memory for input and output data on context1
  for (int i = 0; i < 3; ++i) {
    cudaFree(d_input[i]);
  }
  for (int i = n_input; i < n_io; ++i) {
    delete [] h_output[i - n_input];
    cudaFree(d_output[i - n_input]);
  }
  // destroy stream for context1
  cudaStreamDestroy(stream1);

  // === set input shape for context2 == //
  std::cout << "set input shape for context2" << std::endl;
  int past_seq_length = 6;
  context2->setInputShape(
    "input_ids",
    nvinfer1::Dims2{batch_size, 1}
  );
  context2->setInputShape(
    "position_ids",
    nvinfer1::Dims3{batch_size, 2, 1}
  );
  context2->setInputShape(
    "attention_mask",
    nvinfer1::Dims4{batch_size, 1,  1, 1}
  );
  for (int i = 3; i < n_input; ++i) {
    context2->setInputShape(
      tensor_names[i],
      nvinfer1::Dims4{past_seq_length, batch_size, 32, 128}
    );
  }
  // print input shape for content2
  std::vector<std::size_t> size_list2(n_io);
  std::vector<std::size_t> bytes_list2(n_io);
  for (int i = 0; i < n_io; ++i) {
    // std::cout << "tensor name: " << tensor_names[i] << std::endl;
    auto dims = context2->getTensorShape(tensor_names[i]);
    // std::cout << "tensor shape: ";
    // for (int j = 0; j < dims.nbDims; ++j) {
    //   std::cout << dims.d[j] << ", ";
    // }
    // std::cout << std::endl;
    std::size_t size = 1;
    for (int j = 0; j < dims.nbDims; ++j) {
      size *= dims.d[j];
    }
    // std::cout << "tensor size: " << size << std::endl;
    size_list2[i] = size;
    auto size_result = get_data_type(
      (int)engine->getTensorDataType(tensor_names[i])
    );
    const char * data_type = std::get<0>(size_result);
    std::size_t type_size = std::get<1>(size_result);
    bytes_list2[i] = type_size * size;
    // std::cout << "tensor data type: " << data_type << std::endl;
    // std::cout << "tensor bytes: " << bytes_list2[i] << std::endl;
    // std::cout << std::endl;
  }
  // prepare input data for context2
  int h_input_ids2[] = {19316};
  int h_position_ids2[] = {2, 2};
  bool h_attention_mask2[] = {false};
  int *d_input_ids2;
  int *d_position_ids2;
  bool *d_attention_mask2;
  cudaMalloc((void **)&d_input_ids2, bytes_list2[0]);
  cudaMalloc((void **)&d_position_ids2, bytes_list2[1]);
  cudaMalloc((void **)&d_attention_mask2, bytes_list2[2]);
  CHECK_CUDA(
    cudaMemcpyAsync(
      d_input_ids2,
      h_input_ids2,
      bytes_list2[0],
      cudaMemcpyHostToDevice,
      stream2
    )
  );
  CHECK_CUDA(
    cudaMemcpyAsync(
      d_position_ids2,
      h_position_ids2,
      bytes_list2[1],
      cudaMemcpyHostToDevice,
      stream2
    )
  );
  CHECK_CUDA(
    cudaMemcpyAsync(
      d_attention_mask2,
      h_attention_mask2,
      bytes_list2[2],
      cudaMemcpyHostToDevice,
      stream2
    )
  );
  void * d_input2[59] = {d_input_ids2, d_position_ids2, d_attention_mask2};
  for (int i = 3; i < n_input; ++i) {
    cudaMalloc((void **)&d_input2[i], bytes_list2[i]);
    // 暂时不知道设置啥好，先都设置为1吧
    cudaMemset(d_input2[i], 1, bytes_list2[i]);
  }
  // prepare output data for context2
  __half * h_output2[59];
  void * d_output2[59];
  for (int i = n_input; i < n_io; ++i) {
    h_output2[i - n_input] = new __half[size_list2[i]];
    cudaMalloc((void **)&d_output2[i - n_input], bytes_list2[i]);
  }
  // bind data to context2
  for (int i = 0; i < n_io; ++i) {
    context2->setTensorAddress(
      tensor_names[i], d_input2[i]
    );
  }
  for (int i = n_input; i < n_io; ++i) {
    context2->setTensorAddress(
      tensor_names[i], d_output2[i - n_input]
    );
  }
  // run inference for context2 on stream2
  context2->enqueueV3(stream2);
  cudaDeviceSynchronize();
  cudaStreamSynchronize(stream2);
  // copy output data from device to host
  for (int i = n_input; i < n_io; ++i) {
    CHECK_CUDA(
      cudaMemcpyAsync(
        h_output2[i - n_input],
        d_output2[i - n_input],
        bytes_list2[i],
        cudaMemcpyDeviceToHost,
        stream2
    ));
  }
  cudaDeviceSynchronize();
  cudaStreamSynchronize(stream2);
  std::cout << "output[0] for context2: " << __half2float(h_output2[0][0]) << std::endl;
   
  // free memory for input and output data on context2
  for (int i = 0; i < n_io; ++i) {
    cudaFree(d_input2[i]);
  }
  for (int i = n_input; i < n_io; ++i) {
    delete [] h_output2[i - n_input];
    cudaFree(d_output2[i - n_input]);
  }
  // free memory for context2
  cudaStreamDestroy(stream2);
  return 0;
}



