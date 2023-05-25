#include <kernel.hpp>

void _check_cuda(cudaError_t code, const char *func, const char *file, int line) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << func << " " << file << ":" << line << std::endl;
    throw std::runtime_error("CUDA Error");
  }
}

void Logger::log(Severity severity, const char* msg) noexcept {
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


bool load_engine_data(const std::string engine_path, std::vector<char> & engine_data) {
  std::ifstream engine_file(
    engine_path,
    std::ios::binary | std::ios::in
  );
  engine_file.seekg(0, std::ios::end);
  size_t engine_size = engine_file.tellg();
  // std::cout << "engine_size: " << (engine_size >> 20) << "M" << std::endl;
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


Kernel::Kernel (const std::string engine_path, int batch_size): batch_size_(batch_size) {
  // kernel init
  this->runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
    nvinfer1::createInferRuntime(this->logger_)
  );
  this->load_engine(engine_path);
  this->vertify_io_number();
  this->init_execute_context();
}

Kernel::~Kernel() {
  // kernel free
  std::cout << "free stream1" << std::endl;
  cudaStreamDestroy(this->stream_1_);
  std::cout << "free stream2" << std::endl;
  cudaStreamDestroy(this->stream_2_);
}


void Kernel::load_engine(const std::string engine_path) {
  std::vector<char> engine_data;
  bool result1 = load_engine_data(engine_path, engine_data);
  if (!result1) {
    throw std::runtime_error("Failed to load engine");
  }
  this->engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
    this->runtime_->deserializeCudaEngine(
      engine_data.data(), engine_data.size()
    )
  );
  if (this->engine_ == nullptr) {
    throw std::runtime_error("Faield to deserialize engine");
  }  
}

void Kernel::vertify_io_number() {
  // std::cout << "vertify tensorRT engine io number" << std::endl;
  std::size_t n_io = this->engine_->getNbIOTensors(); 
  if (n_io != this->n_total_) {
    throw std::runtime_error(
      "Number of IO tensors is not correct, must be " \
      + std::to_string(this->n_total_) \
      + ", but you have " \
      + std::to_string(n_io) + " tensors"
    );
  }
  std::size_t n_input = 0;
  std::size_t n_output = 0;
  for (int i = 0; i < n_io; ++i) {
    const char * name = this->engine_->getIOTensorName(i);
    this->tensor_names_.push_back(name);
    if (this->engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      ++n_input;
    } else {
      ++n_output;
    }
  }
  if (n_input != this->n_input_) {
    throw std::runtime_error(
      "Number of input tensors is not correct, must be " \
      + std::to_string(this->n_input_) \
      + ", but you have " \
      + std::to_string(n_input) + " tensors"
    );
  }
  if (n_output != this->n_output_) {
    throw std::runtime_error(
      "Number of output tensors is not correct, must be " \
      + std::to_string(this->n_output_) \
      + ", but you have " \
      + std::to_string(n_output) + " tensors"
    );
  }
  // std::cout << "vertify tensorRT engine io number is OK!" << std::endl;
  // vertify the number of optimization profile
  std::size_t n_profile = this->engine_->getNbOptimizationProfiles();
  if (n_profile != 2) {
    throw std::runtime_error(
      "Number of optimization profiles is not correct, must be 2, but you have " \
      + std::to_string(n_profile) + " profiles"
    );
  }
  std::cout << "number of profile: " << n_profile << std::endl;
}

void Kernel::init_execute_context() {
  CHECK_CUDA(cudaStreamCreate(&this->stream_1_));
  this->context_1_ = std::shared_ptr<nvinfer1::IExecutionContext>(
    this->engine_->createExecutionContext()
  );
  this->context_1_->setOptimizationProfileAsync(0, this->stream_1_);

  // this context is for inference when past_key_values is not None
  CHECK_CUDA(cudaStreamCreate(&this->stream_2_));
  this->context_2_ = std::shared_ptr<nvinfer1::IExecutionContext>(
    this->engine_->createExecutionContext()
  );
  this->context_2_->setOptimizationProfileAsync(1, this->stream_2_);
}

std::vector<std::vector<__half>> Kernel::forward(
  std::vector<int> & input_ids,
  std::vector<int> & position_ids, 
  std::vector<bool> & attention_mask
) {
  // forward for context 1
  if (input_ids.size() % this->batch_size_ != 0) {
    throw std::runtime_error(
      "input_ids.size() % batch_size != 0 "
    );
  }
  if (position_ids.size() % this->batch_size_ != 0) {
    throw std::runtime_error(
      "position_ids.size() % batch_size != 0 "
    );
  }

  if (attention_mask.size() % this->batch_size_ != 0) {
    throw std::runtime_error(
      "attention_mask.size() % batch_size != 0 "
    );
  }
  int seq_length = input_ids.size() / this->batch_size_;
  this->set_input_for_context1(seq_length);
  // std::vector<std::size_t> size_list(this->n_total_);
  std::vector<std::size_t> bytes_list(this->n_total_);
  std::vector<std::size_t> type_bytes_list(this->n_total_);
  this->get_tensor_size(this->context_1_, bytes_list, type_bytes_list);
  std::vector<std::vector<std::vector<__half>>> past_key_values;
  // convert attention_mask to char
  std::vector<char> attention_mask_char(
    attention_mask.begin(), attention_mask.end()
  );
  return std::move(
    this->run_gpu_inference(
      input_ids,
      position_ids,
      attention_mask_char,
      past_key_values, 
      bytes_list,
      type_bytes_list,
      this->context_1_,
      this->stream_1_
    )
  );
}

std::vector<std::vector<__half>> Kernel::forward(
  std::vector<int> & input_ids,
  std::vector<int> & position_ids, 
  std::vector<bool> & attention_mask,
  std::vector<std::vector<std::vector<__half>>> & past_key_values
) {
  // forward for context 2
  // data shape vertification
  if (input_ids.size() % this->batch_size_ != 0) {
    throw std::runtime_error(
      "input_ids.size() % batch_size != 0 "
    );
  }
  if (position_ids.size() % this->batch_size_ != 0) {
    throw std::runtime_error(
      "position_ids.size() % batch_size != 0 "
    );
  }
  if (past_key_values.size() != 28) {
    throw std::runtime_error(
      "past_key_values.size() != 28"
    );
  }
  if (past_key_values[0].size() != 2) {
    throw std::runtime_error(
      "past_key_values.size() != 28"
    );
  }
  if ((past_key_values[0][0].size() / 32 / 128) % this->batch_size_ != 0) {
    throw std::runtime_error(
      "past_key_values.size() % batch_size != 0 "
    );
  }
  
  // set input shape for context 2
  int past_seq_length = past_key_values[0][0].size() / 32 / 128 / this->batch_size_;
  this->set_input_for_context2(past_seq_length);
  // get tensor size for context2
  std::vector<std::size_t> bytes_list(this->n_total_);
  std::vector<std::size_t> type_bytes_list(this->n_total_);
  this->get_tensor_size(this->context_2_, bytes_list, type_bytes_list);
  std::cout << "get tensor size for context 2 ok!" << std::endl;
  // convert attention_mask to char
  std::vector<char> attention_mask_char(
    attention_mask.begin(), attention_mask.end()
  );
  return std::move(
    this->run_gpu_inference(
      input_ids,
      position_ids,
      attention_mask_char,
      past_key_values,
      bytes_list,
      type_bytes_list,
      this->context_2_,
      this->stream_2_
    )
  );
}

    
void Kernel::set_input_for_context1(const int seq_length) {
  this->context_1_->setInputShape(
    "input_ids",
    nvinfer1::Dims2{this->batch_size_, seq_length}
  );
  this->context_1_->setInputShape(
    "position_ids",
    nvinfer1::Dims3{this->batch_size_, 2, seq_length}
  );
  this->context_1_->setInputShape(
    "attention_mask",
    nvinfer1::Dims4{this->batch_size_, 1,  seq_length, seq_length}
  );
}


void Kernel::set_input_for_context2(const int past_seq_length) {
  this->context_2_->setInputShape(
    "input_ids",
    nvinfer1::Dims2{this->batch_size_, 1}
  );
  this->context_2_->setInputShape(
    "position_ids",
    nvinfer1::Dims3{this->batch_size_, 2, 1}
  );
  this->context_2_->setInputShape(
    "attention_mask",
    nvinfer1::Dims4{this->batch_size_, 1,  1, 1}
  );
  for (int i = 3; i < this->n_input_; ++i) {
    this->context_2_->setInputShape(
      this->tensor_names_[i],
      nvinfer1::Dims4{past_seq_length, this->batch_size_, 32, 128}
    );
  }
}

void Kernel::get_tensor_size(
  std::shared_ptr<nvinfer1::IExecutionContext> & context,
  // std::vector<std::size_t> & size_list,
  std::vector<std::size_t> & bytes_list,
  std::vector<std::size_t> & type_bytes_list
) {
  for (int i = 0; i < this->n_total_; ++i) {
    const char * tensor_name = this->tensor_names_[i];
    LOG("tensor name: %s\n", tensor_name);
    auto dims = context->getTensorShape(this->tensor_names_[i]);
    LOG("tensor shape: ");
    for (int j = 0; j < dims.nbDims; ++j) {
      LOG("%d, ", dims.d[j]);
    }
    LOG("\n");
    std::size_t size = 1;
    for (int j = 0; j < dims.nbDims; ++j) {
      size *= dims.d[j];
    }
    // std::cout << "tensor size: " << size << std::endl;
    // size_list[i] = size;
    auto size_result = get_data_type(
      (int)this->engine_->getTensorDataType(this->tensor_names_[i])
    );
    const char * data_type = std::get<0>(size_result);
    std::size_t type_size = std::get<1>(size_result);
    type_bytes_list[i] = type_size;
    bytes_list[i] = type_size * size;
    LOG("tensor bytes: %ld\n", bytes_list[i]);
    LOG("tensor type: %s\n", data_type);
    LOG("\n");
  }
}

std::vector<std::vector<__half>> Kernel::run_gpu_inference(
  const std::vector<int> & input_ids,
  const std::vector<int> & position_ids,
  const std::vector<char> & attention_mask,
  const std::vector<std::vector<std::vector<__half>>> & past_key_values, 
  std::vector<std::size_t> & bytes_list,
  std::vector<std::size_t> & type_bytes_list,
  std::shared_ptr<nvinfer1::IExecutionContext> context, 
  cudaStream_t & stream
) {
  std::vector<std::vector<__half>> h_output(this->n_output_);
  void * d_input[this->n_input_] = {};
  // for input_ids
  CHECK_CUDA(cudaMallocAsync((void **)&d_input[0], bytes_list[0], stream));
  CHECK_CUDA(
    cudaMemcpyAsync(
      d_input[0],
      input_ids.data(),
      bytes_list[0],
      cudaMemcpyHostToDevice,
      stream
    )
  );
  context->setTensorAddress(
      this->tensor_names_[0], d_input[0]
  );
  // for position_ids
  CHECK_CUDA(cudaMallocAsync((void **)&d_input[1], bytes_list[1], stream));
  CHECK_CUDA(
    cudaMemcpyAsync(
      d_input[1],
      position_ids.data(),
      bytes_list[1],
      cudaMemcpyHostToDevice,
      stream
    )
  );
  context->setTensorAddress(
      this->tensor_names_[1], d_input[1]
  );
  // for attention_mask
  CHECK_CUDA(cudaMallocAsync((void **)&d_input[2], bytes_list[2], stream));
  CHECK_CUDA(
    cudaMemcpyAsync(
      d_input[2],
      attention_mask.data(),
      bytes_list[2],
      cudaMemcpyHostToDevice,
      stream
    )
  );
  context->setTensorAddress(
      this->tensor_names_[2], d_input[2]
  );
  // for past_key_values
  if (past_key_values.size() == 0) {
    for (int i = 3; i < this->n_input_; ++i) {
      std::size_t n_bytes  = type_bytes_list[i];
      CHECK_CUDA(cudaMallocAsync((void **)&d_input[i], n_bytes, stream));
      CHECK_CUDA(cudaMemsetAsync(d_input[i], 0, n_bytes, stream));
      context->setTensorAddress(this->tensor_names_[i], d_input[i]);
    }
  } else {
    for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 2; ++j) {
        int index = i * 2 + j + 3;
        std::size_t n_bytes = type_bytes_list[index];
        CHECK_CUDA(cudaMallocAsync((void **)&d_input[index], n_bytes, stream));
        CHECK_CUDA(
          cudaMemcpyAsync(
            d_input[index],
            past_key_values[i][j].data(),
            n_bytes,
            cudaMemcpyHostToDevice,
            stream
          )
        );
        context->setTensorAddress(this->tensor_names_[index], d_input[index]);
      }
    }
  }
  // prepare output data
  void * d_output[this->n_output_];
  for (int i = this->n_input_; i < this->n_total_; ++i) {
    auto n_size = bytes_list[i] / type_bytes_list[i];
    h_output[i - this->n_input_].resize(n_size);
    CHECK_CUDA(
      cudaMallocAsync(
        (void **)&d_output[i - this->n_input_],
        bytes_list[i],
        stream
      )
    );
    context->setTensorAddress(
      this->tensor_names_[i], d_output[i - this->n_input_]
    );
  }
  // run inference
  context->enqueueV3(stream);
  // cudaDeviceSynchronize();
  // cudaStreamSynchronize(stream);
  // copy output data from device to host
  for (int i = this->n_input_; i < this->n_total_; ++i) {
    CHECK_CUDA(
      cudaMemcpyAsync(
        h_output[i - this->n_input_].data(),
        d_output[i - this->n_input_],
        bytes_list[i],
        cudaMemcpyDeviceToHost,
        stream
    ));
  }
  cudaDeviceSynchronize();
  cudaStreamSynchronize(stream);
  // free device memory
  for (int i = 0; i < this->n_input_; ++i) {
    cudaFree(d_input[i]);
  }
  for (int i = this->n_input_; i < this->n_total_; ++i) {
    cudaFree(d_output[i - this->n_input_]);
  }
  return h_output;
}

