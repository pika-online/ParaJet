# Paraformer ONNX GPU 推理引擎


## 特性
模型地址：https://modelscope.cn/models/QuadraV/Seaco_Paraformer_ONNX_GPU
- 面向gpu友好型计算，优化和拆分 onnx graph，大幅提升推理速度（约2000倍）
- 采用fastapi 异步并发设计，支持多路并发转写


## 安装
```shell
# 配置onnxruntime-gpu
conda create -n parajet python=3.8
conda activate parajet
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install cudnn==8.2.1
pip install onnxruntime-gpu==1.14.1
# 其他依赖
pip install -r requirements.txt
```

## 引擎配置


```python
SR = 16000
config = {
        "engine": "mt", # mt: 多线程， mp: 多进程
        "num_workers": 4, # 后台worker数目
        "batch_wait_seconds": 0.5, # 聚合等待时间，越小响应越快，但也会降低推理效率
        # 模型初始化配置
        "instance":{
            "model_dir": "models/asr/parajet",
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
            "use_gpu": True,
            "feed_fixed_shape": [10,30*SR]
        }

    }
```
## 快速上手

这里模拟长音频推理耗时（参考main.py）：
```python
from ParaJet import *

consumer = Consumer(config)
consumer.start()

pcm_data = audio_i2f(read_audio_file('ParaJet/examples/test.wav')) # 80s

k = 100
pcm_data_large = np.concatenate([pcm_data for _ in range(k)]) # 8000s
pcm_seconds = len(pcm_data_large)/16000


with Timer() as t:
    result = recognize(consumer,pcm_data_large,"新声招聘 阿里巴巴")
    # print(result)
print(f"audio_seconds:{pcm_seconds:.3f}, cost:{t.interval:.3f}, speed:{int(pcm_seconds/t.interval)}x")

consumer.stop()

```
耗时信息如下：
```
audio_seconds:7900.000, cost:3.918, speed:2017x
```

### 转写服务
启动
```
uvicorn ParaJet.server:app --host 0.0.0.0 --port 20001
```

客户端请求 (或参考client.py)
```shell
curl -X POST http://localhost:20001/asr/ \
  -F "file=@ParaJet/examples/test.wav;type=application/octet-stream" \
  -F "hotwords=心森招聘 阿里巴巴"
```
结果如下：
```shell
{"status":"success","msg":null,"result":{"asr":[["嗯",5.749],["那",5.848],["么",6.008],["今",6.148],["天",6.307],["我",6.507],["们",6.647],["就",6.886],["简",7.226],["单",7.365],["的",7.545],["进",7.745],["行",7.904],["一",8.144],["下",8.263],["那",8.483],["个",8.683],["新",8.962],["声",9.222],["招",9.501],["聘",9.76],["的",10.06],["嗯",10.699],["讨",11.397],["论",11.617],["吧",11.856],["因",12.116],["为",12.196],["现",12.395],["在",12.495],["不",12.695],["是",12.854],["好",13.533],["像",13.752],["就",13.872],["新",14.092],["生",14.291],["到",14.491],["校",14.711],["嘛",15.01],["然",15.529],["后",15.729],["我",15.848],["们",15.928],["社",16.048],["团",16.208],["呢",16.447],["也",16.567],["需",16.766],["要",16.906],["招",17.066],["聘",17.265],["一",17.465],["些",17.565],["新",17.725],["的",17.924],["社",18.144],["员",18.403],["然",19.122],["后",19.301],["就",19.481],["今",19.8],["天",19.96],["就",20.14],["大",20.379],["概",20.559],["就",20.739],["讨",20.958],["论",21.158],["一",21.377],["下",21.517],["嗯",22.176],["怎",22.435],["么",22.595],["招",22.735],["聘",22.954],["的",23.194],["内",23.453],["容",23.593],["吧",23.832],["嗯",24.411],["我",24.731],["们",24.85],["就",24.97],["首",25.25],["先",25.389],["想",25.709],["一",25.908],["下",26.048],["那",26.188],["个",26.347],["招",26.766],["聘",27.006],["的",27.186],["地",27.325],["点",27.485],["在",27.784],["哪",27.964],["里",28.144],["吧",28.263],["嗯",29.441],["地",29.721],["天",29.82],["就",29.98]]...},"completed":null,"stream":false}
```
