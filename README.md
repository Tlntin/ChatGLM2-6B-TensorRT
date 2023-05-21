## 使用教程
### 准备工作
1. pytorch->onnx, 这个阶段需要40-64G左右内存。
2. onnx->tensorRT, 这个阶段需要大量显存(目测大概16G-22G左右)以及大量内存（大概70G)，所以显卡要求最低24G显存，推荐32G显存, 内存不够的可以加swap。
3. TensorRT推理阶段，比原版高一些，大概14-17G左右。
4. 安装好了docker与nvidia-docker（可选, 建议搞一个，这样可以节省配环境的时间）
5. 下载huggingface的ChatGLM-6b的权重到本项目根目录，然后将`-`替换为`_`即可, 这一步是为了方便debug。
```bash
git clone https://huggingface.co/THUDM/chatglm-6b.git
mv chatglm-6b chatglm_6b
```
6. 关于tensorRT环境。
- 选择1：本机安装，开发环境使用，目前推荐最新的TensorRT8.6.1+cuda11.8+pytorch2.0
- 选择2：Docker安装，生产环境使用，可以用英伟达官方容器，用下面这个命令直接拉镜像。
```bash
# 拉镜像
docker pull nvcr.io/nvidia/pytorch:23.04-py3

# 临时进入容器（退出后容器自动关闭）
docker run --gpus all \
	-it --rm \
	--ipc=host \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	-v ${PWD}:/workspace/ \
	nvcr.io/nvidia/pytorch:23.04-py3
```

7. 安装依赖
```bash
pip install -r requirements.txt
```

### 第一步：将pytorch导出成onnx
1. 修改chatGLM模型结构，单独改`chatglm_6b/modeling_chatglm.py` 就行了。
- 去除没必要的tensor，将1000行的`layer_id=tensor(i)`改成`layer_id=i`即可，这个是layer_id后续调用也是用的普通数字，没必要改成tensor，改完后可以减少两个警告。

-  修改236行左右的代码，将cos.squeeze(1)和sin.squeeze(1)换成flatten（[参考链接](https://github.com/NVIDIA/TensorRT/issues/2849#issuecomment-1543334514))。因为sueeeze(1)会固定past_key_value的第一个shape,导致推理时，只能输入固定shape的数据。
- 修改前长这样：
```python
cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
        F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
```
- 修改后长这样：
```python
cos, sin = F.embedding(
        position_id, 
        torch.flatten(cos, start_dim=1, end_dim=2) 
    ).unsqueeze(2), \
        F.embedding(
            position_id, 
            torch.flatten(sin, start_dim=1, end_dim=2)
        ).unsqueeze(2)
```
- 当然，如果你偷懒也不是不行，我拿5月19日最新的代码改好放在onnx_export目录了，你可以直接copy过去覆盖就行了, 如果后续官方有更新代码，或许我这个就不适用了，最好还是自己改官方最新代码比较好。。
```bash
cp onnx_export/modeling_chatglm.py chatglm_6b/modeling_chatglm.py 
```
2. 进行onnx_export目录
```bash
cd onnx_export
```

3. 执行export2onnx.py文件。
```bash
python3 export2onnx.py
```
- 问题来了：这个输入输出怎么来的？
- 回答：其实我们加载的模型是`configuration_chatglm.py`的`ChatGLMForConditionalGeneration`类。虽然python版貌似执行的model.chat(),但是实际核心还是pytorch的model(xxx)，也就是调用的forward方法。所以可以直接在`ChatGLMForConditionalGeneration`类的forward方法那里打断点，就可以获取文件的输入参数了。foword函数大概在1174行，可以将断点打在最后的1220行，`if not return_dict`这里。然后调用原版的chat函数，进入debug模式，就可以看到函数的输入参数是啥了。注意：这个入参有两种情况，一种是past_key_values为None,一种是28x2个(past_seq_len, batch_size, 32, 128)的tensor。这里我取的是前一种进行导出onnx, 其实后面一种也可以,为了兼容tensort,我past_key_values将None数值替换成了28x2个`torch.zeros(0, 1, 32, 128)`
- 为啥opset_version选择18？因为最近onnx/tensorRT支持了layerNorm的实现，这个最低要求是17,而目前最高就是18/19,所以我选择18，当然你也可以选择17或者19试试。

4. 验证onnx文件
- 因为我们导出onnx的时候，input_ids的shape是[1, 4], 为了验证其他的shape是否ok,我就编造了一个shape为[1, 5] input_id进行输入，观察模型结构是否正常。
```bash
python3 run_onnx.py
```
- 因为上文说到，`ChatGLMForConditionalGeneration`类的forward方法存在两种情况，一个是past_key_values为None一个不为None,为了验证past_key_values不为None的情况，我们需要执行下面这个文件，如果没报错，说明模型结构正常。
```bash
python3 run_onnx2.py
```
- 一般来说，还需要做pytorch和onnx输出情况差异对比，由于本次仅导出了fp32的文件，一般问题不大，所以这里省略该步骤。
4. 返回上层目录
```bash
cd ..
```


### 第二步，onnx转tensorRT
1. 进入tensorrt_export目录
```bash
cd tensorrt_export
```
2. 将onnx转成TensorRT文件。
```bash
python3 onnx2trt.py
```
- 注意：由于存在两种输入情况，所以这里定义了两个profile来控制输入。

3. 验证tensorRT文件输入输出结构
```bash
python3 read_trt_profile.py
```

4. 返回上一层目录
```bash
cd ..
```


### 第三步，推理
1. 目前已经有大佬开源了推理脚本，你可以直接用现成的。你只需要做将它提供的编译好的tensorRT文件，换成你编译好的tensorRT文件所在路径即可，目前导出的TensorRT路径在项目的`models/chatglm6b-bs1.plan`路径。
- 大佬开源的推理脚本地址：[地址](https://huggingface.co/TMElyralab/lyraChatGLM), 该推理脚本配套使用教程：[地址](https://www.http5.cn/index.php/archives/19/)


### 待做（画饼）
- [ ] 自己实现一个推理方案，支持python/c++
- [ ] 将FastTransformer编译为tensorRT的一个插件，以实现更快的加速方案。


### 其他
- 貌似自tensorRT8.6.1开始，同一个tensorRT版本，不同架构/算力，可以用同一份编译好的tensorRT Engine,也就是说，如果有人用TensorRT8.6.1编译好了共享给你的话，前两步理论上可以省略（不过只支持30系/40系，并且存在一定精度损失）。


### 参考链接
- https://github.com/THUDM/ChatGLM-6B
- https://huggingface.co/TMElyralab/lyraChatGLM
- https://github.com/K024/chatglm-q





