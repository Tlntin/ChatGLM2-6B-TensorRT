## 使用教程
### 准备工作
1. pytorch->onnx, 这个阶段需要40-64G左右内存, 24G以上显存（可选，推荐有）, 推荐32G显存。
2. onnx->tensorRT, 这个阶段需要大量显存(目测大概17G左右)以及大量内存（大概70G)，所以显卡要求最低24G显存,推荐32G显存， 内存不够的可以加swap。
3. TensorRT推理阶段，基本和原版一样，13G左右。
4. 安装好了docker与nvidia-docker（可选, 建议搞一个，这样可以节省配环境的时间）
5. 下载huggingface的ChatGLM-6b的权重到本项目根目录，然后将`-`替换为`_`即可, 这一步是为了方便debug。
```bash
git clone https://huggingface.co/THUDM/chatglm-6b.git
mv chatglm-6b chatglm_6b
```
6. 关于tensorRT环境。
- 最低版本要求：8.6.0
    1. 选择1：pypi安装
    2. 选择2（推荐）：本机安装，开发环境使用，目前推荐最新的TensorRT8.6.1+cuda11.8+cudnn8.9.x+pytorch2.0, cuda用.run格式包安装，cudnn和TensorRT用tar压缩包安装。
        - 安装cudnn示例
        ```bash
        tar -xvf cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tar.xz
        cd cudnn-linux-x86_64-8.9.1.23_cuda11-archive
        sudo cp -r lib/* /usr/local/cuda/lib64/
        sudo cp -r include/* /usr/local/cuda/include/
        ```
        - 安装tensorRT示例
        ```bash
        tar -zxvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
        cd TensorRT-8.6.1.6
        sudo cp bin/* /usr/local/cuda/bin
        sudo cp -r lib/* /usr/local/cuda/lib64
        sudo cp include/* /usr/local/cuda/include

        # 安装TensorRT中附带的python包
        cd python
        # 我的是python3.10,选择python3.10对应的python包
        pip install tensorrt-8.6.1-cp310-none-linux_x86_64.whl
        pip install tensorrt_dispatch-8.6.1-cp310-none-linux_x86_64.whl
        pip install tensorrt_lean-8.6.1-cp310-none-linux_x86_64.whl

        # 返回上一级目录
        cd ..

        # 进入onnx_graphsurgeon
        cd onnx_graphsurgeon
        pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl

        # 返回上一级目录
        cd ../..

        ```
        - 刷新lib库
        ```bash
        sudo ldconfig
        ```

        - 确认cudnn可以被搜索到
        ```bash
        sudo ldconfig -v | grep libcudnn

        # 搜索结果长下面这样
        # libcudnn.so.8 -> libcudnn.so.8.9.1
        # ...
        ```

        - 确认TensorRT库的nvinfer可以被搜索到
        ```bash
        sudo ldconfig -v | grep libnvinfer

        # 搜索结果长下面这样
        # libnvinfer.so.8 -> libnvinfer.so.8.6.1
        # libnvinfer_plugin.so.8 -> libnvinfer_plugin.so.8.6.1
        # ...
        ```

    3. 选择3：Docker安装，生产环境使用，可以用英伟达官方容器，用下面这个命令直接拉镜像。
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
 8. 推荐使用cuda 11.8以及以上，结合lazy load技术，可以加快推理以及节省显存。[链接](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading)
 - 使用方法
 ```bash
 # 建议写到~/.bashrc
 export CUDA_MODULE_LOADING=LAZY
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

3. 执行export2onnx.py文件
```bash
# 强烈推荐
# for GPU显存 >= 24G, 该操作会利用GPU导出fp16的onnx文件, 相对来说更为推荐这个, 如果是刚好24G的显卡，可能会爆显存。
python export2onnx_fp16.py

# 这个导出的onnx比较大，并且转成的tensorRT也会很大
# for GPU显存 < 24G, 该操作会利用CPU导出fp32格式的onnx
python3 export2onnx_fp32.py

```
- 问题：这个输入输出怎么来的？
- 回答：其实我们加载的模型是`configuration_chatglm.py`的`ChatGLMForConditionalGeneration`类。虽然python版貌似执行的model.chat(),但是实际核心还是pytorch的model(xxx)，也就是调用的forward方法。所以可以直接在`ChatGLMForConditionalGeneration`类的forward方法那里打断点，就可以获取文件的输入参数了。foword函数大概在1174行，可以将断点打在最后的1220行，`if not return_dict`这里。然后调用原版的chat函数，进入debug模式，就可以看到函数的输入参数是啥了。注意：这个入参有两种情况，一种是past_key_values为None,一种是28x2个(past_seq_len, batch_size, 32, 128)的tensor。这里我取的是前一种进行导出onnx, 其实后面一种也可以,为了兼容tensort,我past_key_values将None数值替换成了28x2个`torch.zeros(0, 1, 32, 128)`
- 问题：opset_version选择18？
- 回答：因为最近onnx/tensorRT支持了layerNorm的实现，这个最低要求是17,而目前最高就是18/19,所以我选择18，当然你也可以选择17或者19试试。

4. 验证onnx文件(目前fp16导出的onnx验证不通过，需要修改, fp32正常)
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
5. 编译TensorRT C++测试文件，测量pytorch与tensorRT最大精度误差（可选，推荐）。
- 注意：编译inference_test需要安装libtorch, 去官网下载安装即可，需要下载xxx版本而不是pre_cxx版本。
- 正式编译
```bash
mkdir build && cd build
cmake ..
make

# 执行
./inference_test

```

### 第三步，推理
1. 运行这个即可，只是一个简单的测试(需要gcc/g++ >= 9 并且安装了cuda/TensorRt C++版)
```bash
python demo.py
```

### 速度测试
1. 硬件平台：
- CPU: i9-10900k
- GPU: nvidia 3090(24G)
- 内存：64G金士顿3200

2. 原版 fp16 batch_size=1 速度：27-29token/s
3. TensorRT fp16 batch_size=1 速度：39-42token/s
4. 综合提速：34.4%-55.5%




### 待做（画饼）
- [x] 自己实现一个推理方案，支持python/c++
- [ ] 将FastTransformer编译为tensorRT的一个插件，以实现更快的加速方案。

### 参考链接
- https://github.com/THUDM/ChatGLM-6B
- https://huggingface.co/TMElyralab/lyraChatGLM
- https://github.com/K024/chatglm-q
