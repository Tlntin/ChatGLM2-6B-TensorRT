import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import os
import numpy as np


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
onnx_path = os.path.join(project_dir, "output", "onnx_output", "chatglm_6b.onnx")
new_onnx_path = os.path.join(project_dir, "output", "new_onnx_output", "chatglm_6b.onnx")
# 加载 ONNX 模型
onnx_model = onnx.load(onnx_path)

# 获取所有节点名称
node_names = set(node.name for node in onnx_model.graph.node)

# 获取所有输入和输出张量名称
input_names = set(input.name for input in onnx_model.graph.input)
output_names = set(output.name for output in onnx_model.graph.output)

# 将所有节点标记为输出
for node in onnx_model.graph.node:
    for output in node.output:
        # 如果输出张量不是输入张量，则将其标记为输出
        if output not in input_names:
            onnx_model.graph.output.append(helper.make_tensor_value_info(output, TensorProto.UNDEFINED, None))

# 保存修改后的 ONNX 模型
onnx.save(
  onnx_model, 
  new_onnx_path,
  save_as_external_data=True,
  all_tensors_to_one_file=False
)
sess = ort.InferenceSession(new_onnx_path, providers=["CPUExecutionProvider"])
input_feed = {
    "input_ids": np.random.randint(0, 100, size=(1, 512)).astype(np.int64),
    "position_ids": np.random.randint(0, 100, size=(1, 2, 512)).astype(np.int64),
    "attention_mask": np.random.randint(0, 1, size=(1, 1, 512, 512)).astype(np.bool_),
    "past_key_values.0.decoder.key": np.random.randint(0, 100, size=(1, 1, 32, 128)).astype(np.float32),
    "past_key_values.0.decoder.value": np.random.randint(0, 100, size=(1, 1, 32, 128)).astype(np.float32),
}

output = sess.run(None, input_feed)
print(output)