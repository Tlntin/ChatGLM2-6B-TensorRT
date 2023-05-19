import onnx
import os

# Load the ONNX model
model = onnx.load("onnx_output/chatglm_6b.onnx")
for node in model.graph.node:
  # if node.name in ["If_92", "If_100"]:
  if node.op_type == "If":
    # print(node)
    attr_name = node.attribute[1].name
    output1 = node.attribute[0].g.output[0]
    output2 = node.attribute[1].g.output[0]
    output_dim1 = output1.type.tensor_type.shape.dim
    output_dim2 = output2.type.tensor_type.shape.dim
    # print(len(output_dim1))
    # print(len(output_dim2))
    if len(output_dim1) != len(output_dim2):
        print("======old node======")
        print(node)
        attr_name = node.attribute[1].name
        output_name = node.attribute[1].g.output[0].name
        sub_graph_name = node.attribute[1].g.name
        node.attribute[1].CopyFrom(node.attribute[0])
        node.attribute[1].g.name = sub_graph_name
        node.attribute[1].name = attr_name
        node.attribute[1].g.output[0].name = output_name
        node.attribute[1].g.node[-1].output[0] = output_name
        node.attribute[1].g.node[0].output[0] = "my_" + node.attribute[1].g.node[0].output[0]
        node.attribute[1].g.node[1].input[1] = "my_" + node.attribute[1].g.node[1].input[1]
        for n in node.attribute[1].g.node:
            n.name = "my_" + n.name
        print("======new node======")
        print(node)
# Create an external data directory to save tensors
external_data_dir = "onnx_output2"
if not os.path.exists(external_data_dir):
    os.makedirs(external_data_dir)
else:
    for file in os.listdir(external_data_dir):
        os.remove(os.path.join(external_data_dir, file))

# Save the model with external data
# not support onnx memory > 2GB, to sovel it, see this url: https://github.com/onnx/onnx/issues/3275
onnx.save(
    model,
    "onnx_output2/chatglm_6b.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=False
)
print("onnx saved success")