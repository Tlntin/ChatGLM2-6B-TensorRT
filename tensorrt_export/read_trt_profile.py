import tensorrt as trt

# Load TensorRT Engine
with open("models/chatglm6b-bs1.plan", 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# print input and output
print("Inputs and Output:")
for i in range(engine.num_bindings):
    tensor_name = engine.get_tensor_name(i)
    tensor_mode = engine.get_tensor_mode(tensor_name)
    tensor_shape = engine.get_tensor_shape(tensor_name)
    if tensor_mode == trt.TensorIOMode.INPUT:
        print(f"input_name:'{tensor_name}', input_shape: {tensor_shape}", )
    else:
        print(f"output_name:'{tensor_name}', input_shape: {tensor_shape}", )


# print profile
# print("Profile configuration:")
# for i in range(engine.num_optimization_profiles):
#     print(" - profile index: ", i)
#     for j in range(engine.num_bindings):
#         tensor_name = engine.get_tensor_name(j)
#         tensor_mode = engine.get_tensor_mode(tensor_name)
#         tensor_shape = engine.get_tensor_shape(tensor_name)
#         if tensor_mode == trt.TensorIOMode.INPUT:
#             print("- input name: ", tensor_name)
#             print("min|opt|max shape: ", engine.get_tensor_profile_shape(tensor_name, i))
#             # print("min|opt|max shape: ", engine.get_profile_shape(i, j))

print("Profile configuration:")
for i in range(engine.num_optimization_profiles):
    print(" - profile index: ", i)
    for j in range(engine.num_bindings):
        if engine.binding_is_input(j):
            profile = engine.get_profile_shape(i, j)
            print("   - input name: ", engine.get_binding_name(j))
            print("     min shape: ", profile[0])
            print("     max shape: ", profile[1])
            print("     optimal shape: ", profile[2])
