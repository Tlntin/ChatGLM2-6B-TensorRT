import tensorrt as trt
import os
import time
from colored import fg, stylize
# import onnx
from polygraphy.backend.trt import network_from_onnx_path
from itertools import tee
from tensorrt import MemoryPoolType, PreviewFeature

# default is 1, maybe you can try 2, 4, 8, 16
batch_size = 1
max_length  = 2048
opt_length = max_length
# if use force use fp16, may reduce the accuracy and memory usage
force_use_fp16 = False
# default 3, max 5, 5 is the best but need more GPU memory and time
builder_optimization_level = 3
# lower memory GPU can try this option with True \
# it can use CPU memory/CPU compute to run some layers, but may reduce the speed
all_gpu_fallback = False

if batch_size > 1 and builder_optimization_level != 5:
    raise Exception("batch size > 1, please use builder_optimization_level = 5")


class MyLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        if severity == trt.Logger.ERROR:
            print(stylize("[ERROR] " + msg, fg("red")))  # 红色字体
        elif severity == trt.Logger.WARNING:
            print(stylize("[WARNING] " + msg, fg("yellow")))  # 黄色字体
        elif severity == trt.Logger.INTERNAL_ERROR:
            print(stylize("[INTERNAL_ERROR] " + msg, fg("red")))  # 红色字体
        elif severity == trt.Logger.INFO:
            print(stylize("[INFO] " + msg, fg("green")))  # 绿色字体
        # elif severity == trt.Logger.VERBOSE:
        #     print(stylize("[VERBOSE] " + msg, fg("blue")))  # 蓝色字体
        # else:
        #     print("[UNKNOWN] " + msg)


def get_network_profiles(trt_builder):
    # ----profile1 when past_key_values is None----
    profile1 = trt_builder.create_optimization_profile()
    profile2 = trt_builder.create_optimization_profile()
    profile1.set_shape(
        "input_ids",
        (1, 1),
        (batch_size, opt_length),
        (batch_size, max_length),
    )
    profile1.set_shape(
        "position_ids",
        (1, 2, 1),
        (batch_size, 2, opt_length),
        (batch_size, 2, max_length),  
    )
    profile1.set_shape(
        "attention_mask",
        (1, 1, 1, 1),
        (batch_size, 1, opt_length, opt_length),
        (batch_size, 1, max_length, max_length),
    )
    for layer_idx in range(28):
        input_names = [
            f"past_key_values.{layer_idx}.decorder.key",
            f"past_key_values.{layer_idx}.decorder.value"
        ]
        for name in input_names:
            profile1.set_shape(
                name,
                (0, 1, 32, 128),
                (0, batch_size, 32, 128),
                (0, batch_size, 32, 128),
            )
    # ----profile2 when past_key_values is not None----
    profile2.set_shape(
        "input_ids",
        (1, 1),
        (batch_size, 1),
        (batch_size, 1),
    )
    profile2.set_shape(
        "position_ids",
        (1, 2, 1),
        (batch_size, 2, 1),
        (batch_size, 2, 1),
    )
    profile2.set_shape(
        "attention_mask",
        (1, 1, 1, 1),
        (batch_size, 1, 1, 1),
        (batch_size, 1, 1, 1),
    )
    for layer_idx in range(28):
        input_names = [
            f"past_key_values.{layer_idx}.decorder.key",
            f"past_key_values.{layer_idx}.decorder.value"
        ]
        for name in input_names:
            profile2.set_shape(
                name,
                (1, 1, 32, 128),
                (opt_length - 1, batch_size, 32, 128),
                (max_length - 1, batch_size, 32, 128),
            )
    profiles = [profile1, profile2]
    return profiles


def get_network_definition(trt_network):
    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    
    num_layers = trt_network.num_layers 
    indices = list(range(num_layers))
    for i, i_next in pairwise(indices):
        layer = trt_network.get_layer(i)
        l_next = trt_network.get_layer(i_next)

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
    return trt_network

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
onnx_path = os.path.join(project_dir,"output", "onnx_output", "chatglm_6b.onnx")
model_dir = os.path.join(project_dir, "models")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
tensorrt_engine_path = os.path.join(model_dir, f"chatglm6b-bs{batch_size}.plan")
builder = trt.Builder(MyLogger())
builder.max_threads = os.cpu_count() // 2
config = builder.create_builder_config()
profile_list = get_network_profiles(builder)
for profile in profile_list:
    config.add_optimization_profile(profile)
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
config.set_memory_pool_limit(MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)

preview_features = [
    PreviewFeature.PROFILE_SHARING_0806,
    # PreviewFeature.FASTER_DYNAMIC_SHAPES_0805,
    # PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805
]
for feature in preview_features:
    config.set_preview_feature(feature, True)
config.builder_optimization_level = builder_optimization_level


# load onnx model
print("loading onnx model from ", onnx_path)
network =  builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
# trt_parser = trt.OnnxParser(network, builder.logger)
# onnx_model = onnx.load(onnx_path)
# trt_parser.parse(onnx_model)

_, network, _ = network_from_onnx_path(onnx_path)

network = get_network_definition(network)
print("=============tensorRT inference config =====================")
if builder_optimization_level == 3:
    print("==tensorRT engine begin compile, maybe you need wait 10-25 minute ==")
elif builder_optimization_level == 5:
    print("==tensorRT engine begin compile, maybe you need wait 30-60 minute ==")
else:
    print("==tensorRT engine begin compile, maybe you need wait a moment ==")
    
# trt_engine = engine_from_network(
#     (trt_builder, network, onnx_parser),
#     trt_inference_config
# )
serialized_engine = builder.build_serialized_network(network, config)
# 保存引擎到文件
with open(tensorrt_engine_path, "wb") as f:
    f.write(serialized_engine)
# save_engine(trt_engine, tensorrt_engine_path)
print("==tensorRT engine compile done==")
# rename tensorRT engine file with file size
print("wait 10 senconds")
time.sleep(10)
file_size = round(os.path.getsize(tensorrt_engine_path) / (1024 ** 3), 1)
print(f"tensorRT engine file size is {file_size}G")
new_trt_path = os.path.join(
    model_dir, f"chatglm6b-bs{batch_size}-{file_size}G.plan"
)
os.rename(tensorrt_engine_path, new_trt_path)
print("tensorRT engine save to ", new_trt_path)