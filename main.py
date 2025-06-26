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
