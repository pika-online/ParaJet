from parajet import *

consumer = Consumer(config)
consumer.start()

k = 100
pcm_data = audio_i2f(read_audio_file(r'parajet/examples/test.wav')) # 80s
pcm_data = np.concatenate([pcm_data for _ in range(k)]) # 8000s
pcm_seconds = len(pcm_data)/16000


with Timer() as t:
    result = recognize(consumer,pcm_data,"")

text = "".join([item[0] for item in result])
print(text)
print(f"audio_seconds:{pcm_seconds:.3f}, cost:{t.interval:.3f}, speed:{int(pcm_seconds/t.interval)}x")

consumer.stop()
