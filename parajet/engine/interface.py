from parajet.utils import *

class Consumer:
    
    def __init__(self,config:dict):
        
        self.config = config
        self.engine = config['engine']
        if self.engine == 'mt':
            self.inputs1 = queue.Queue()
            self.inputs2 = queue.Queue()
            self.outputs = queue.Queue()
        elif self.engine == 'mp':
            self.inputs1 = multiprocessing.Queue()
            self.inputs2 = multiprocessing.Queue()
            self.outputs = multiprocessing.Queue()

    
        self.num_workers = config["num_workers"]
        self.batch_wait_seconds = config["batch_wait_seconds"]
        self.B, self.T = config['instance']["feed_fixed_shape"]

        self.stop_event = threading.Event()
        self.thread_backend_aggregation = threading.Thread(target=self._backend_aggregation,daemon=True)
        self.thread_backend_summary = threading.Thread(target=self._backend_summary,daemon=True)
        self.workers = []
        self.results = {}
        self.lock = threading.Lock()
        
    def _now(self):
        return time.time()

    @staticmethod
    def _init_session(config):
        from parajet.core.interface import SeacoParaformer
        session = SeacoParaformer(
            model_dir=config['instance']['model_dir'],
            intra_op_num_threads=config['instance']['intra_op_num_threads'],
            inter_op_num_threads=config['instance']['inter_op_num_threads'],
            use_gpu=config['instance']["use_gpu"],
            fixed_shape=config['instance']["feed_fixed_shape"]
        )
        return  session

    
    @staticmethod
    def worker_process(id,cls,config,task_queue, result_queue, feed_fixed_shape):
        print(f'开始进程:{id}')

        session = cls._init_session(config)
        result_queue.put('ready')
        print('完成模型加载')

        B,T = feed_fixed_shape
        while True:
            try:
                item = task_queue.get()
            except:
                continue
            
            if item is None:
                break 

            taskIds, waveforms, configs = item
            size = len(taskIds)
            hotwords = [config['hotwords'] for config in configs]

            # 整理成 fixed_shape_input
            taskIds_fixed = ['' for _ in range(B)]
            waveforms_fixed = np.zeros((B,T),dtype='float32')
            hotwords_fixed = ['' for _ in range(B)]
            for i in range(size):
                taskIds_fixed[i] = taskIds[i]
                waveforms_fixed[i] = waveforms[i]
                hotwords_fixed[i] = hotwords[i]

    
            result = session(waveforms_fixed, hotwords_fixed)

            print(f"推理耗时：{result['time_cost']}, 稀疏度: {size}/{B}")
            for taskId,asr in zip(taskIds_fixed,result['asr']):
                if taskId:
                    result_queue.put((taskId,asr))


        print(f'退出进程:{id}')



    # 任务聚合线程
    def _backend_aggregation(self):

        print("[START]: _backend_aggregation")

        a,b,c,d = [],[],[],[] # 聚合变量
        while not self.stop_event.is_set():

            
            # 限定时间内抓取batch
            while 1:
                try:
                    item = self.inputs1.get(timeout=self.batch_wait_seconds)
                    timestamp, taskId, waveform, config = item
                    a.append(timestamp)
                    b.append(taskId)
                    c.append(waveform)
                    d.append(config)
                except queue.Empty:
                    break

            # 处理batch
            while len(a):
                _a,a = a[:self.B], a[self.B:]
                _b,b = b[:self.B], b[self.B:]
                _c,c = c[:self.B], c[self.B:]
                _d,d = d[:self.B], d[self.B:]
                self.inputs2.put((_b,_c,_d))


        print("[END]: _backend_aggregation")


    # 结果整合线程
    def _backend_summary(self):

        print("[START]: _backend_summary")
        while not self.stop_event.is_set():
            try:
                item = self.outputs.get(timeout=1e-3)
                taskId, result = item 
                with self.lock:
                    self.results[taskId].put(result)
            except queue.Empty:
                continue

        print("[END]: _backend_summary")


    def submit(self,taskId,input_data,config):
        self.inputs1.put((time.time(),taskId,input_data,config))
        if taskId not in self.results:
            self.results[taskId] = queue.Queue()

    def get(self,taskId,timeout=None):
        res = self.results[taskId].get(timeout=timeout)
        if not self.results[taskId].qsize():
            with self.lock:
                del self.results[taskId]
        return res



    def start(self):
        cls = self.__class__
        for i in range(self.num_workers):
            if self.engine=='mt':
                p = threading.Thread(target=self.worker_process,args=(i,cls,self.config,self.inputs2,self.outputs,[self.B,self.T]),daemon=True)
            if self.engine=='mp':
                p = multiprocessing.Process(target=self.worker_process,args=(i,cls,self.config,self.inputs2,self.outputs,[self.B,self.T]),daemon=True)
            p.start()
            self.workers.append(p)
        for i in range(self.num_workers):
            self.outputs.get()

        self.thread_backend_aggregation.start()
        self.thread_backend_summary.start()
        print('启动成功')
        
        

    def stop(self):
        self.stop_event.set()
        for i in range(self.num_workers):
            self.inputs2.put(None)
        for worker in self.workers:
            worker.join()
        self.thread_backend_aggregation.join()
        self.thread_backend_summary.join()
        print('关闭成功')


if __name__ == "__main__":

    from ..config import config

    consumer = Consumer(config)
    consumer.start()

    pcm_data = audio_i2f(read_audio_file('parajet/examples/test.wav'))
    pcm_data = reshape_audio_to_BxT(pcm_data, consumer.T)
    n = pcm_data.shape[0]

    all_seconds = sum([len(x) for x in pcm_data])/16000
    hotwords = ["心森招聘 阿里巴巴" for _ in range(n)]

    with Timer() as t:
        tasks = []
        for segment,hotword in zip(pcm_data,hotwords):
            taskId = generate_random_string(20)
            consumer.submit(taskId,segment,{'hotwords':'hotword'})
            tasks.append(taskId)
        
        for taskId in tasks:
            result = consumer.get(taskId)
            print(result)
    
    print(all_seconds,t.interval,all_seconds/t.interval)
    consumer.stop()

