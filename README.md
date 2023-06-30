## 使用教程
### 准备工作: pytorch->ONNX
1. pytorch->onnx, 这个阶段需要40-64G左右内存（如果需要导出fp16格式的onnx，需要24G以上显存，推荐32G显存）。
2. 安装依赖，注意：pytorch转ONNX需要，pytorch需要大于等于2.1.0，所以需要安装nightly版**
```bash
pip install -r requirements.txt
```
### 准备工作2： ONNX->TensorRT
1. onnx->tensorRT, 这个阶段需要大量显存(目测大概16G左右)以及大量内存（大概70G)，所以显卡要求最低16G显存,推荐24G显存， 内存不够的可以加swap。
2. TensorRT推理阶段，基本和原版一样，12.5G左右。
3. 安装好了docker与nvidia-docker（可选, 建议搞一个，这样可以节省配环境的时间）
4. 下载huggingface的ChatGLM-6b的权重到本项目根目录，然后将`-`替换为`_`即可, 这一步是为了方便debug。
```bash
git lfs install
git clone --depth=1 https://huggingface.co/THUDM/chatglm2-6b
mv chatglm2-6b chatglm2_6b
```
5. 关于tensorRT环境。
- 最低版本要求：8.6.0, 推荐8.6.1
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


6. 推荐使用cuda 11.8以及以上，结合lazy load技术，可以加快推理以及节省显存。[链接](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading)
 - 使用方法
 ```bash
 # 建议写到~/.bashrc
 export CUDA_MODULE_LOADING=LAZY
 ```

### 第一步：将pytorch导出成onnx
1. 导出ONNX模型
- for GPU显存 > 24G
- 该操作会利用GPU导出fp16的onnx文件, 导出后可以用run_onnx_cuda.py来校准一下精度看看是否。
```bash
# 导出模型
python onnx_export/export2onnx.py --data_type=fp16

```

- for GPU显存 < 24G，或者无GPU用户（推荐）
- 该操作会利用CPU导出fp32格式的onnx, 导出后可以用`run_onnx_cpu.py`和`run_onnx_cpu2.py`校准精度。
```bash
# 导出模型 
python3 export2onnx.py

```
2. 检查模型
- 检查模型输入输出
```bash
polygraphy inspect model output/onnx_output/chatglm2_6b.onnx
```
- 上面命令运行后输出如下
```bash
==== ONNX Model ====
    Name: torch_jit | ONNX Opset: 18
    
    ---- 58 Graph Input(s) ----
    {input_ids [dtype=int64, shape=('batch_size', 'sequence')],
     position_ids [dtype=int64, shape=('batch_size', 'sequence')],
     past_key_values.0.key [dtype=float32, shape=('past_sequence', 'batch_size', 2, 128)],
```
- 可以看出好像attention_mask不见了，两种可能。第一种就是你的attention_mask没有给到输入参数，第二种可能是attention_mask可以不输入，为None值也不影响结果，所以Onnx自动给优化掉了。
- 经过大量测试，我排除了第一种可能。所以我需要试试第二种可能，对比attention_mask为None时，是否影响原结果。
- 运行下面这个代码对比一下。
```bash
python3 onnx_export/export_test.py 
```
- 实验发现，attention_mask为None时，对结果没有影响，所以可以认为导出ONNX时优化attention_mask是OK的。
- 再尝试一下将attention_mask设置为空Tensor,对结果也没有影响
```bash
python3 onnx_export/export_test_v2.py
```
- 顺便再尝试一下第一次推理时，将past_key_values设置为0shape,看看是否有影响。
```bash
python3 onnx_export/export_test_v3.py
```
- 测试没有影响，说明onnx推理/TensorRT推理的时候，第一次forward可以将past_key_values设置为0 shape。
- 既然attention_mask对输出结果没有影响，可以在导出onnx的时候直接将attention_mask干掉
- 这里需要做一些修改，将`chatglm2_6b/modeling_chatglm.py`文件中的`ChatGLMForConditionalGeneration`类中的forward方法中的`attention_mask: Optional[torch.Tensor] = None`这个参数剔除（大概在914行），
- 修改前
```python
def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
):
      ....
```
- 修改后
```python
def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
):
    attention_mask: Optional[torch.Tensor] = None
```
- 修改完的`modeling_chatglm.py`我已经放到`onnx_export`目录，仅供参考
- 然后再执行一次导出onnx, 这次完全将`attention_mask`剔除
```bash
python3 onnx_export/export2onnx_v2.py 
```

3. 对比输入/输出数据
- for fp16（还没测试，没有大显存）
```bash
# 准备校准的数据
python3 onnx_export/export_compare_data.py --data_type=fp16

# 测试cuda上面推理结果
python3 onnx_export/run_onnx_cuda.py
```
- for fp32
```bash
# 准备校准的数据
python3 onnx_export/export_compare_data.py

# 测试cpu上面的推理结果
python3 onnx_export/run_onnx_cpu.py
```


### 第二步，onnx转tensorRT（还未彻底完工，待续）
1. 检查onnx文件，观察是否存在TensorRT不兼容算子。
```bash
polygraphy inspect capability output/onnx_output/chatglm2_6b.onnx
```
- 提示"Graph is fully supported by TensorRT; Will not generate subgraphs."则说明没有问题。

2. 将带cache的onnx转成TensorRT
- 直接用python来转,此时输入参数较多，用trtexec不太方便。
- `>`这个是用来重定向输出到某个文件，如果是Windows，则直接用前面的运行py文件命令即可。
```bash
python3 tensorrt_export/onnx2trt_with_cache.py > trt_with_past.log 2>&1 
```

4. 验证tensorRT文件输入输出结构
```bash
polygraphy inspect model models/chatglm6b2-bs1_with_cache.plan > models/with_cache.log 2>&1
```

5. 检查数据精度，验证TensorRT文件输出结果和pytorch是否一样（目前测试来看误差有点大，暂时没办法用）。
```bash
python3 tensorrt_export/trt_check_with_past.py 
python3 tensorrt_export/trt_check_no_past.py 
```
- 经检测误差较大，需要优化一下，通过一些算子来合并Onnx

### ~~~第三步，推理~~

1. ~~运行这个即可，只是一个简单的测试(需要gcc/g++ >= 9 并且安装了cuda/TensorRt C++版)~~
```bash
python demo.py
```

### ~~速度测试~~
1. ~~硬件平台：~~
- ~~CPU: i9-10900k~~
- ~~GPU: nvidia 3090(24G)~~
- ~~内存：64G金士顿 DDR4 3200~~

2. ~~原版 fp16 batch_size=1 速度：27-29token/s~~
3. ~~TensorRT fp16 batch_size=1 速度：39-42token/s~~
4. ~~综合提速：34.4%-55.5%~~~~




### 待做（画饼）
- [x] 自己实现一个推理方案，支持python/c++
- [ ] 将FastTransformer编译为tensorRT的一个插件，以实现更快的加速方案。

### 参考链接
- https://github.com/THUDM/ChatGLM-6B
- https://huggingface.co/TMElyralab/lyraChatGLM
- https://github.com/K024/chatglm-q
