import tensorrt as trt
import os
from itertools import tee

from polygraphy.backend.trt import (
    network_from_onnx_path,
    engine_from_network,
    save_engine,
    Profile,
)

from polygraphy.backend.trt import CreateConfig
from tensorrt import PreviewFeature, MemoryPoolType


batch_size = 1
max_length  = 512
opt_length = max_length // 2



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
    
    indices = list(range(0, network_definition[1].num_layers))
    for i, i_next in pairwise(indices):
        layer = network_definition[1].get_layer(i)
        l_next = network_definition[1].get_layer(i_next)

        if not all([layer.get_output(i).is_execution_tensor for i in range(layer.num_outputs)]):
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

    return network_definition


input_fpath = "../onnx_output/chatglm_6b.onnx"


preview_features = [PreviewFeature.FASTER_DYNAMIC_SHAPES_0805]



trt_inference_config = CreateConfig(
                tf32=False,
                fp16=True,
                memory_pool_limits = {MemoryPoolType.WORKSPACE: 2 * 1024 * 1024 * 1024},
                profiles=profiles,
                precision_constraints=("obey"),
                preview_features=preview_features
            )


onnx_network = network_from_onnx_path(input_fpath)


network_definition = get_network_definition(onnx_network)
print(network_definition)
print("=============tensorRT inference config =====================")
print(trt_inference_config)
print("==tensorRT engine begin compile, maybe you need wait 10-25 minute ==")
trt_engine = engine_from_network(network_definition, trt_inference_config)
print(trt_engine)

if not os.path.exists("../models"):
    os.mkdir("../models")

output_fpath = "../models/chatglm6b-bs1.plan"
save_engine(trt_engine, output_fpath)