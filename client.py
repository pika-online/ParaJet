import requests

audio_file = "ParaJet/examples/test.wav"
with open(audio_file,'rb') as f:
    auido_bytes = f.read()
hotwords = "新声招聘 阿里巴巴"

# 设置服务地址
BASE_URL = "http://localhost:20001/asr/"  # 修改为你的服务器地址

files = {
    "file": ("audio", auido_bytes, "application/octet-stream")
}
data = {
    "hotwords": hotwords
}

response = requests.post(BASE_URL, files=files, data=data, headers={})

data = response.json()
print(data)