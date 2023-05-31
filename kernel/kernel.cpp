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
  if (std::filesystem::exists(engine_path) == false || std::filesystem::is_regular_file(engine_path) == false) {
    std::cerr << RED << "Failed to find engine file" << NONE << std::endl;
    return false;
  }
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


Kernel::Kernel (const std::string engine_path, int batch_size) : batch_size_(batch_size) {
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
  std::cout << GREEN << "INFO: free stream1" << NONE << std::endl;
  cudaStreamDestroy(this->stream_1_);
  std::cout << GREEN << "INFO: free stream2" << NONE << std::endl;
  cudaStreamDestroy(this->stream_2_);
}


void Kernel::load_engine(const std::string engine_path) {
  std::vector<char> engine_data;
  bool result1 = load_engine_data(engine_path, engine_data);
  if (!result1) {
    throw std::runtime_error(
      "Failed to load engine, " + engine_path + " not found!"
    );
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
  std::cout << GREEN << "number of profile: " << n_profile << NONE << std::endl;
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

std::vector<torch::Tensor> Kernel::forward(
  std::vector<torch::Tensor> & input_tensors
) {
  // check batch size
  /*
  // input_tensors[0] is input_ids
  if (input_tensors[0].size(0) != this->batch_size_ ) {
    throw std::runtime_error(
      "input_ids batch size is not correct, must be " \
      + std::to_string(this->batch_size_)
    );
  }
  // input_tensors[1] is position_ids
  if (input_tensors[1].size(0) != this->batch_size_) {
    throw std::runtime_error(
      "position_ids batch size is not correct, must be " \
      + std::to_string(this->batch_size_)
    );
  }
  */
  const int seq_len = input_tensors[0].size(1);
  int present_key_len = 0;
  // temp code to test the inference with empty past_key_values
  int input_size = this->n_input_;
  torch::Device device = input_tensors[0].device();
  std::vector<std::size_t> bytes_list(this->n_total_);
  std::vector<std::size_t> type_bytes_list(this->n_total_);
  std::shared_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream;
  if (input_tensors.size() == 3) {
    /*
    // input_tensors[2] is attention_mask
    if (input_tensors[2].size(0) != this->batch_size_) {
      throw std::runtime_error(
        "attention_mask batch size is not correct, must be " \
        + std::to_string(this->batch_size_)
      );
    }
    */
    int seq_length = input_tensors[0].size(1);
    this->set_input_for_context1(seq_length);
    // std::vector<std::size_t> size_list(this->n_total_);
    
    this->get_tensor_size(this->context_1_, bytes_list, type_bytes_list);
    present_key_len = seq_len;
    // fill the empty tensor to d_input
    for (int i = 3; i < this->n_input_; ++i) {
      torch::Tensor empty_tensor = torch::zeros(
        {1},
        torch::dtype(torch::kFloat16)
      ).to(device);
      input_tensors.push_back(std::move(empty_tensor));
    }
    context = this->context_1_;
    stream = this->stream_1_;
  } else {
    /*
    if (past_key_values.size() != 28) {
      throw std::runtime_error(
        "past_key_values.size() != 28"
      );
    }
    if (past_key_values[0].size() != 2) {
      throw std::runtime_error(
        "past_key_values[0].size() != 2"
      );
    }
    if (past_key_values[0][0].size(1) != this->batch_size_) {
      throw std::runtime_error(
        "past_key_values[0][0] batch_size is not correct, must be " \
        + std::to_string(this->batch_size_)
      );
    }
    */
    // set input shape for context 2
    int past_seq_length = input_tensors[3].size(0);
    // std::cout << "past_seq_length: " << past_seq_length << std::endl;
    this->set_input_for_context2(past_seq_length);
    // get tensor size for context2
    this->get_tensor_size(this->context_2_, bytes_list, type_bytes_list);
    MY_LOG("get tensor size for context 2 ok!\n");
    present_key_len = seq_len + past_seq_length;
    context = this->context_2_;
    stream = this->stream_2_;
  }

  // prepare output
  std::vector<torch::Tensor> output_tensors;
  MY_LOG("seq len: %d\n", seq_len);
  MY_LOG("present_key_len: %d\n", present_key_len);
  // output for present_key
  for (int i = 0; i < this->n_output_ - 1; ++i) {
    output_tensors.push_back(
      std::move(
        torch::zeros(
          {present_key_len, this->batch_size_, 32, 128}, 
          torch::dtype(torch::kFloat16)
        ).to(device)
      )
    );
  }
  // output for logists
  output_tensors.push_back(
    std::move(
      torch::zeros(
        {this->batch_size_, seq_len, this->vocab_size_},
        torch::dtype(torch::kFloat16)
      ).to(device)
    )
  );

  this->run_gpu_inference(
    input_tensors,
    output_tensors,
    context,
    stream
  );
  return std::move(output_tensors);
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
  MY_LOG("======= get tensor size ====");
  for (int i = 0; i < this->n_total_; ++i) {
    const char * tensor_name = this->tensor_names_[i];
    MY_LOG("tensor name: %s\n", tensor_name);
    auto dims = context->getTensorShape(this->tensor_names_[i]);
    MY_LOG("tensor shape: ");
    for (int j = 0; j < dims.nbDims; ++j) {
      MY_LOG("%d, ", dims.d[j]);
    }
    MY_LOG("\n");
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
    MY_LOG("tensor bytes: %ld\n", bytes_list[i]);
    MY_LOG("tensor type: %s\n", data_type);
    MY_LOG("\n");
  }
  MY_LOG("======= get tensor size ok ====");
  MY_LOG("\n");
}

void Kernel::run_gpu_inference(
  const std::vector<torch::Tensor> & input_tensors,
  std::vector<torch::Tensor> & output_tensors,
  std::shared_ptr<nvinfer1::IExecutionContext> & context, 
  cudaStream_t & stream
) {
  MY_LOG("=== prepare run gpu interference =\n");
  std::vector<void *> empty_ptr_list;
  // set input
  for (int i = 0; i < this->n_input_; ++i) {
    context->setTensorAddress(
      this->tensor_names_[i],
      input_tensors[i].data_ptr()
    );

  }
  // set output
  for (int i = this->n_input_; i < this->n_total_; ++i) {
    context->setTensorAddress(
      this->tensor_names_[i],
      output_tensors[i - this->n_input_].data_ptr()
    );
  }
  // run inference
  context->enqueueV3(stream);
  cudaDeviceSynchronize();
  cudaStreamSynchronize(stream);
}

