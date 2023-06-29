import tensorrt as trt
import os
import time
from colored import fg, stylize
import json
# import onnx
from polygraphy.backend.trt import network_from_onnx_path
from itertools import tee
from tensorrt import MemoryPoolType, PreviewFeature

now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = os.path.join(project_dir, "output")
time_cache_path = os.path.join(output_dir, "fp16_no_cache.cache")

# default is 1, maybe you can try 2, 4, 8, 16
batch_size = 1
use_time_cache = True
max_length = 2048
opt_length = max_length // 2
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
        elif severity == trt.Logger.VERBOSE:
            print(stylize("[VERBOSE] " + msg, fg("blue")))  # 蓝色字体
        else:
            print("[UNKNOWN] " + msg)


def get_network_profiles(trt_builder, num_layers=28):
    # ----profile1 when past_key_values is None----
    profile1 = trt_builder.create_optimization_profile()
    profile1.set_shape(
        "input_ids",
        (1, 1),
        (batch_size, opt_length),
        (batch_size, max_length),
    )
    profile1.set_shape(
        "position_ids",
        (1, 1),
        (batch_size, opt_length),
        (batch_size, max_length),
    )
    profiles = [profile1]
    return profiles


def get_network_definition(trt_network):
    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    layer_type_set = set() 
    num_layers = trt_network.num_layers
    indices = list(range(num_layers))
    for i, i_next in pairwise(indices):
        layer = trt_network.get_layer(i)
        next_layer = trt_network.get_layer(i_next)
        if not all([
            layer.get_output(i).is_execution_tensor
            for i in range(layer.num_outputs)
        ]):
            continue
        if layer.get_output_type(0) != trt.float32:
            continue
        layer_type_set.add(str(layer.type))
        if layer.type == trt.LayerType.ELEMENTWISE and \
                next_layer.type == trt.LayerType.REDUCE:
            layer.__class__ = getattr(trt, "IElementWiseLayer")
            if layer.op == trt.ElementWiseOperation.POW:
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)

            next_layer.precision = trt.float32
            next_layer.set_output_type(0, trt.float32)
        #else:
        #    layer.precision = trt.DataType.HALF
    layer_type_path = os.path.join(output_dir, "layer_type.json")
    with open(layer_type_path, "wt") as f1:
        json.dump(list(layer_type_set), f1, indent=4)
    return trt_network


if __name__ == "__main__":
    onnx_path = os.path.join(
        output_dir, "onnx_output_no_cache", "chatglm2_6b.onnx"
    )
    model_dir = os.path.join(project_dir, "models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    tensorrt_engine_path = os.path.join(
        model_dir, f"chatglm6b2-bs{batch_size}_no_cache.plan"
    )
    builder = trt.Builder(MyLogger())
    builder.max_threads = os.cpu_count() // 2
    config = builder.create_builder_config()
    profile_list = get_network_profiles(builder)
    for profile in profile_list:
        config.add_optimization_profile(profile)
    # use fp16
    # config.flags = 1 << int(trt.BuilderFlag.FP16)
    # disable tf32
    config.flags = config.flags & ~(1 << int(trt.BuilderFlag.TF32))
    # use obey precision constraints
    config.flags = config.flags | (1 << int(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS))
    # config.set_memory_pool_limit(MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
    # use preview features
    preview_features = [
        # PreviewFeature.PROFILE_SHARING_0806,
        PreviewFeature.FASTER_DYNAMIC_SHAPES_0805,
        # PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805
    ]
    for feature in preview_features:
        config.set_preview_feature(feature, True)
    config.builder_optimization_level = builder_optimization_level

    # use time cache
    time_cache = b""
    # read time cache
    if use_time_cache:
        if os.path.exists(time_cache_path):
            time_cache = open(time_cache_path, "rb").read()
            if time_cache is None:
                time_cache = b""
                print(stylize("read time cache failed", fg("red")))
            else:
                print(stylize(f"read time cache from {time_cache_path}", fg("green")))
        else:
            time_cache = b""
            print(stylize("time cache will init with empty.", fg("green")))

        # set time cache
        cache = config.create_timing_cache(time_cache)
        config.set_timing_cache(cache, False)

    # load onnx model
    print("loading onnx model from ", onnx_path)
    # network =  builder.create_network(
    #     1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # )
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

    if serialized_engine is not None:
        # 保存引擎到文件
        with open(tensorrt_engine_path, "wb") as f:
            f.write(serialized_engine)
        # save_engine(trt_engine, tensorrt_engine_path)
        print("==tensorRT engine compile done==")
    else:
        raise RuntimeError("build engine failed")

    # save time cache
    if use_time_cache and not os.path.exists(time_cache_path):
        time_cache = config.get_timing_cache()
        if time_cache is not None:
            time_cache_data = time_cache.serialize()
            open(time_cache_path, "wb").write(time_cache_data)
            print(
                stylize(
                    "save time cache to {}".format(time_cache_path),
                    fg("green")
                )
            )