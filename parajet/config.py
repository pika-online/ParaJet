SR = 16000
config = {
        "engine": "mt", # mt: 多线程， mp: 多进程
        "num_workers": 1, # 后台worker数目
        "batch_wait_seconds": 0.5, # 聚合等待时间，越小响应越快，但也会降低推理效率
        # 模型初始化配置
        "instance":{
            "model_dir": "models/parajet",
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
            "use_gpu": True,
            "feed_fixed_shape": [10,30*SR]
        }

    }