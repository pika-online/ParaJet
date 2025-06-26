from .engine.interface import Consumer
from .utils import *
from .config import config

def recognize(consumer:Consumer,pcm_data,hotwords=""):
    
    segment_seconds = consumer.T/16000
    samples = reshape_audio_to_BxT(pcm_data,consumer.T)
    print(f"音频({len(pcm_data)}) -> Reshape: {samples.shape}")
    tasks = []
    for sample in  samples:
        taskId = generate_random_string(20)
        consumer.submit(taskId,sample,{'hotwords':hotwords})
        tasks.append(taskId)

    result = []
    for i,taskId in enumerate(tasks):
        for token, ts in consumer.get(taskId):
            ts2 = i * segment_seconds + ts 
            result.append((token,ts2)) 
    return result
