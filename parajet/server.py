from fastapi import FastAPI, UploadFile, Request, File, Form, Query
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from parajet import Consumer, recognize
from parajet.utils import *



# FastAPI 配置
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源访问，生产环境建议设置为特定域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有请求方法（GET、POST、PUT等）
    allow_headers=["*"],  # 允许所有请求头
)

from .config import config
consumer = Consumer(config)
consumer.start()


# 回复体
def make_response(status,msg,result={},completed=None,stream=False,string=False):
    response = {
        "status":status,
        "msg":msg,
        "result":result,
        "completed": completed,
        'stream':stream
    }
    
    if status =='error' and stream:
        response["completed"] = True
        
    if string:
        response = json.dumps(response)
    return response

@app.post("/asr/")
async def asr(
    file: UploadFile = File(...),
    hotwords: str = Form("魔塔社区 阿里巴巴"),
):
    try:
        content = {}

        # 读取 PCM 音频数据
        audio_bytes = await file.read()
        pcm_data = audio_i2f(read_audio_bytes(audio_bytes))

        # 语音识别
        asr_result = await asyncio.to_thread(recognize,consumer=consumer,pcm_data=pcm_data,hotwords=hotwords)
        # asr_result = "".join([x[0] for x in asr_result])
        content['asr'] = asr_result

        return make_response(status="success",msg=None,result=content)
    
    except Exception as e:
        return make_response(status="error",msg=f"ERROR:{e}",result=None)