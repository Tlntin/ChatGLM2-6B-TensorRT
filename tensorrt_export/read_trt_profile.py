import tensorrt as trt
import os


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
model_dir = os.path.join(project_dir, "models")
trt_enginge_path = os.path.join(model_dir, "chatglm6b-bs1-18.5G.plan")

# Load TensorRT Engine
with open(trt_enginge_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# print input and output
print("Inputs and Output:")
n_io = engine.num_io_tensors
tensor_names = []
tensor_modes = []
for i in range(n_io):
    tensor_name = engine.get_tensor_name(i)
    tensor_names.append(tensor_name)
    tensor_mode = engine.get_tensor_mode(tensor_name)
    tensor_modes.append(tensor_mode)
    tensor_shape = engine.get_tensor_shape(tensor_name)
    if tensor_mode == trt.TensorIOMode.INPUT:
        print(f"input_name:'{tensor_name}', input_shape: {tensor_shape}", )
    else:
        print(f"output_name:'{tensor_name}', input_shape: {tensor_shape}", )


print("Profile configuration:")
for i in range(engine.num_optimization_profiles):
    print('=' * 20) 
    print(" - profile index: ", i)
    for j in range(n_io):
        tensor_name = tensor_names[j]
        tensor_mode = tensor_modes[j]
        if tensor_mode == trt.TensorIOMode.INPUT:
            # profile = engine.get_profile_shape(i, j)
            profile = engine.get_tensor_profile_shape(tensor_name, i)
            print("   - input name: ", tensor_name)
            print("     min shape: ", profile[0])
            print("     opt shape: ", profile[1])
            print("     max shape: ", profile[2])
