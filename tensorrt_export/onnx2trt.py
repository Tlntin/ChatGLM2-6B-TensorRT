import tensorrt as trt
import os
import time
from polygraphy.backend.trt import (
    network_from_onnx_path,
    engine_from_network,
    save_engine,
    Profile,
)
from itertools import tee
from polygraphy.backend.trt import CreateConfig
from tensorrt import MemoryPoolType, PreviewFeature

# default is 1, maybe you can try 2, 4, 8, 16
batch_size = 1
max_length  = 512
opt_length = max_length // 2
# if use force use fp16, may reduce the accuracy and memory usage
force_use_fp16 = False
# default 3, max 5, 5 is the best but need more GPU memory and time
builder_optimization_level = 5
# lower memory GPU can try this option with True \
# it can use CPU memory/CPU compute to run some layers, but may reduce the speed
all_gpu_fallback = False

if batch_size > 1 and builder_optimization_level != 5:
    raise Exception("batch size > 1, please use builder_optimization_level = 5")



# ----profile1 when past_key_values is None----
profile1 = Profile().add(
    "input_ids",
    min=(1, 1),
    opt=(batch_size, opt_length),
    max=(batch_size, max_length),
).add(
       "position_ids",
    min=(1, 2, 1),
    opt=(batch_size, 2, opt_length),
    max=(batch_size, 2, max_length),  
).add(
        "attention_mask",
    min=(1, 1, 1, 1),
    opt=(batch_size, 1, opt_length, opt_length),
    max=(batch_size, 1, max_length, max_length),
)
for layer_idx in range(28):
    input_names = [
        f"past_key_values.{layer_idx}.decorder.key",
        f"past_key_values.{layer_idx}.decorder.value"
    ]
    for name in input_names:
        profile1.add(
            name,
            min=(0, 1, 32, 128),
            opt=(0, batch_size, 32, 128),
            max=(0, batch_size, 32, 128),
        )
# ----profile2 when past_key_values is not None----
profile2 = Profile().add(
    "input_ids",
    min=(1, 1),
    opt=(batch_size, 1),
    max=(batch_size, 1),
).add(
    "position_ids",
    min=(1, 2, 1),
    opt=(batch_size, 2, 1),
    max=(batch_size, 2, 1),
).add(
    "attention_mask",
    min=(1, 1, 1, 1),
    opt=(batch_size, 1, 1, 1),
    max=(batch_size, 1, 1, 1),
)
for layer_idx in range(28):
    input_names = [
        f"past_key_values.{layer_idx}.decorder.key",
        f"past_key_values.{layer_idx}.decorder.value"
    ]
    for name in input_names:
        profile2.add(
            name,
            min=(1, 1, 32, 128),
            opt=(opt_length - 1, batch_size, 32, 128),
            max=(max_length - 1, batch_size, 32, 128),
        )

profiles = [profile1, profile2]


def get_network_definition(network_definition):
    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    
    num_layers = network_definition[1].num_layers 
    indices = list(range(num_layers))
    for i, i_next in pairwise(indices):
        layer = network_definition[1].get_layer(i)
        l_next = network_definition[1].get_layer(i_next)

        if not all([
            layer.get_output(i).is_execution_tensor
            for i in range(layer.num_outputs)
        ]):
            continue

        if layer.num_outputs > 0 and layer.get_output_type(0) != trt.float32:
            continue

        if layer.type == trt.LayerType.ELEMENTWISE and l_next.type == trt.LayerType.REDUCE:
            layer.__class__ = getattr(trt, "IElementWiseLayer")
            if layer.op == trt.ElementWiseOperation.POW:
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)

            l_next.precision = trt.float32
            l_next.set_output_type(0, trt.float32)
            continue
        # need GPU memory 16G
        if not force_use_fp16:
            continue
        for i in range(layer.num_outputs):
            if layer.get_output_type(0) == trt.float32: 
                layer.precision = trt.float16
                if layer.type != trt.LayerType.CAST:
                    layer.set_output_type(i, trt.float16)
    return network_definition

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
input_fpath = os.path.join(project_dir, "onnx_output", "chatglm_6b.onnx")
model_dir = os.path.join(project_dir, "models")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
tensorrt_engine_path = os.path.join(model_dir, f"chatglm6b-bs{batch_size}.plan")
preview_features = [
    PreviewFeature.PROFILE_SHARING_0806,
    PreviewFeature.FASTER_DYNAMIC_SHAPES_0805,
    PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805
]



trt_inference_config = CreateConfig(
                tf32=False,
                fp16=True,
                memory_pool_limits = {MemoryPoolType.WORKSPACE: 2 * 1024 * 1024 * 1024},
                profiles=profiles,
                precision_constraints="obey",
                # precision_constraints="prefer",
                preview_features=preview_features,
                #sparse_weights=True,
                profiling_verbosity=trt.ProfilingVerbosity.DETAILED,
                builder_optimization_level=builder_optimization_level,
                allow_gpu_fallback=all_gpu_fallback,
            )

print("loading onnx model from ", input_fpath)
onnx_network = network_from_onnx_path(input_fpath)


network_definition = get_network_definition(onnx_network)
print(network_definition)
print("=============tensorRT inference config =====================")
print(trt_inference_config)
if builder_optimization_level == 3:
    print("==tensorRT engine begin compile, maybe you need wait 10-25 minute ==")
elif builder_optimization_level == 5:
    print("==tensorRT engine begin compile, maybe you need wait 30-60 minute ==")
else:
    print("==tensorRT engine begin compile, maybe you need wait a moment ==")
    
trt_engine = engine_from_network(network_definition, trt_inference_config)
print(trt_engine)

save_engine(trt_engine, tensorrt_engine_path)
print("==tensorRT engine compile done==")
# rename tensorRT engine file with file size
print("wait 10 senconds")
time.sleep(10)
file_size = round(os.path.getsize(tensorrt_engine_path) / (1024 ** 3), 1)
print(f"tensorRT engine file size is {file_size}G")
new_trt_path = os.path.join(
    model_dir, f"chatglm6b-bs{batch_size}_{file_size}G.plan"
)
os.rename(tensorrt_engine_path, new_trt_path)
print("tensorRT engine save to ", new_trt_path)