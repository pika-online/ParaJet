from parajet.lib import *


def read_audio_file(audio_file):
    """读取音频文件数据并转换为PCM格式。"""
    ffmpeg_cmd = [
        FFMPEG,
        '-i', audio_file,
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', '16k',
        '-ac', '1',
        'pipe:']
    with subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False) as proc:
        stdout_data, stderr_data = proc.communicate()
    pcm_data = np.frombuffer(stdout_data, dtype=np.int16)
    return pcm_data


def read_audio_bytes(audio_bytes):
    ffmpeg_cmd = [
    FFMPEG,
    '-i', 'pipe:',  
    '-f', 's16le',
    '-acodec', 'pcm_s16le',
    '-ar', '16k',
    '-ac', '1',
    'pipe:' ]
    with subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False) as proc:
        stdout_data, stderr_data = proc.communicate(input=audio_bytes)
    pcm_data = np.frombuffer(stdout_data, dtype=np.int16)
    return pcm_data

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

def audio_f2i(data, width=16):
    """将浮点数音频数据转换为整数音频数据。"""
    data = np.array(data)
    return np.int16(data * (2 ** (width - 1)))

def audio_i2f(data, width=16):
    """将整数音频数据转换为浮点数音频数据。"""
    data = np.array(data)
    return np.float32(data / (2 ** (width - 1)))

def generate_random_string(n):
    letters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(letters) for i in range(n))
    return random_string

def reshape_audio_to_BxT(audio: np.ndarray, T: int) -> np.ndarray:
    """
    将音频reshape为 B x T，长度不足T补零，超过T分块（不足一块的尾部补零）
    
    参数:
        audio: 原始音频数据，1D numpy 数组
        T: 每块的时间长度（采样点数）
    
    返回:
        reshaped_audio: shape 为 (B, T) 的 numpy 数组
    """
    L = len(audio)
    B = int(np.ceil(L / T))  # 计算所需的块数
    padded_length = B * T
    padded_audio = np.zeros(padded_length, dtype=audio.dtype)
    padded_audio[:L] = audio  # 补零
    
    reshaped_audio = padded_audio.reshape(B, T)
    return reshaped_audio
