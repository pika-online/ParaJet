from .front_end import *
from .utils import *
from .ort_session import OrtInferSession
from ParaJet.utils import *

class SeacoParaformer:
    def __init__(self,
                 model_dir,
                 intra_op_num_threads=1,
                 inter_op_num_threads=1,
                 use_gpu=True,
                 fixed_shape=[10, 16000]) -> None:
        # 加载模型配置
        with open(f"{model_dir}/config.yaml", "r", encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

        # 加载分词信息
        with open(f"{model_dir}/tokens.json", "r", encoding="utf-8") as f:
            self.token_list = json.load(f)

        with open(f"{model_dir}/seg_dict", "r", encoding="utf-8") as f:
            lines = [line.strip().split() for line in f.readlines()]
            self.sentence_piece = {line[0]: line[1:] for line in lines}

        # 构建 token 映射
        self.token2id = {token: i for i, token in enumerate(self.token_list)}
        self.id2token = {i: token for i, token in enumerate(self.token_list)}

        # CMVN 和前端参数
        self.cmvn = load_cmvn(f"{model_dir}/am.mvn")
        self.lfr_m = self.config['frontend_conf']['lfr_m']
        self.lfr_n = self.config['frontend_conf']['lfr_n']
        self.sr = 16000

        # ONNX 模型加载
        self.model_eb = OrtInferSession(f"{model_dir}/model_eb.onnx", intra_op_num_threads, inter_op_num_threads, use_gpu)
        self.model_encoder = OrtInferSession(f"{model_dir}/encoder_and_predictor_v2.onnx", intra_op_num_threads, inter_op_num_threads, use_gpu)
        self.model_decoder = OrtInferSession(f"{model_dir}/decoder.onnx", intra_op_num_threads, inter_op_num_threads, use_gpu)

        # 热词参数和嵌入初始化
        self.hotword_sos_embedding = self.model_eb([np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)])[0][0][0]
        self.hotword_max_len = 10
        self.hotword_max_nums = 30

        # 最大音频长度（秒）和 token 数
        self.max_audio_seconds = fixed_shape[1] / self.sr
        self.max_token_nums = int(self.max_audio_seconds * 5)

        # 预热模型
        dummy_waveforms = np.zeros(fixed_shape, dtype='float32')
        dummy_hotwords = [''] * fixed_shape[0]
        self.__call__(dummy_waveforms, dummy_hotwords)

        self.running = False

    def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[int]]:
        # 提取 FBank 特征，应用 LFR 和 CMVN
        waveform_tensor = torch.tensor(np.array(waveform_list, dtype='float32'))
        feats = extract_fbank_batch_torch(waveform_tensor).numpy()
        
        feats_processed, feats_len = [], []
        for feat in feats:
            feat_lfr = apply_lfr(feat, self.lfr_m, self.lfr_n)
            feat_cmvn = apply_cmvn(feat_lfr, self.cmvn).astype('float32')
            feats_processed.append(feat_cmvn)
            feats_len.append(len(feat_cmvn))

        return feats_processed, feats_len


    def process_hotwords(self, highlight: List[str]) -> np.ndarray:
        """处理热词输入，生成热词嵌入张量"""
        batch_size = len(highlight)
        hotword_task = {i: [] for i in range(batch_size)}

        for i, sentence in enumerate(highlight):
            for word in sentence.split():
                if '\u4e00' <= word[0] <= '\u9fff':  # 中文按字分割
                    tokens = list(word)
                else: # 英文查表
                    tokens = self.sentence_piece.get(word, [])

                token_ids = [self.token2id.get(token, 8403) for token in tokens]
                word_len = len(token_ids)
                if word_len > self.hotword_max_len:
                    print(f"热词“{word}”长度超过限制 {self.hotword_max_len}")
                    continue
                token_ids_padded = token_ids + [0] * (self.hotword_max_len - word_len)

                hotword_task[i].append([word_len - 1, token_ids_padded, None])

        # 批量嵌入
        all_tokens = [item[1] for sublist in hotword_task.values() for item in sublist]
        if all_tokens:
            embeddings = self.model_eb([np.array(all_tokens, dtype='int32')])[0].transpose(1, 0, 2)
            idx = 0
            for i in hotword_task:
                for j in range(len(hotword_task[i])):
                    l = hotword_task[i][j][0]
                    hotword_task[i][j][-1] = embeddings[idx][l]
                    idx += 1

        # 构造最终输入 B x N x 512
        hotword_inputs = np.array([[self.hotword_sos_embedding] * (self.hotword_max_nums + 1) for _ in range(batch_size)])
        for i in range(batch_size):
            for j, (_, _, embedding) in enumerate(hotword_task[i]):
                if j > self.hotword_max_nums:
                    print(f"热词数超过限制 {self.hotword_max_nums}")
                    break
                hotword_inputs[i, j] = embedding

        return hotword_inputs

    def __call__(self, waveforms: list, highlight: list):
        self.running = True
        assert len(waveforms) == len(highlight)

        result = {'asr': None, 'time_cost': {}}

        with Timer() as t:
            hotword_inputs = self.process_hotwords(highlight)
        result['time_cost']['hotword'] = f"{t.interval:.3f}"

        # 特征提取
        with Timer() as t:
            feats, feats_len = self.extract_feat(waveforms)
        result['time_cost']['fbank'] = f"{t.interval:.3f}"

        # 编码器推理
        with Timer() as t:
            us_alphas, alpha, token_num_float, encode_out = self.model_encoder([feats])
        result['time_cost']['encoder'] = f"{t.interval:.3f}"

        # CIF 层
        with Timer() as t:
            padded_frames, token_nums = cif(encode_out, alpha, max_len=self.max_token_nums)
        result['time_cost']['cif'] = f"{t.interval:.3f}"

        # 解码器推理
        with Timer() as t:
            logits = self.model_decoder([encode_out, hotword_inputs, padded_frames])[0]
        result['time_cost']['decoder'] = f"{t.interval:.3f}"

        # 取最大概率的 token ID
        token_ids = logits.argmax(axis=-1)

        # 还原为文本及时间戳
        tmp = []
        with Timer() as t:
            for token_id, token_num, us_alpha in zip(token_ids, token_nums, us_alphas):
                tokens = [self.id2token[i] for i in token_id][:token_num]
                alignment = cif2(tokens.copy(), us_alpha, audio_seconds=self.max_audio_seconds)
                tmp.append(alignment)
        result['asr'] = tmp
        result['time_cost']['timestamp'] = f"{t.interval:.3f}"

        result['time_cost']['all'] = f"{sum(map(float, result['time_cost'].values())):.3f}"
        self.running = False
        return result

# 测试入口
if __name__ == "__main__":

    B, T = 20, 16000 * 30  # 定义GPU模型的吞吐能力
    model = SeacoParaformer(
        model_dir="models/asr/parajet",
        intra_op_num_threads=1,
        inter_op_num_threads=1,
        use_gpu=True,
        fixed_shape=[B, T]
    )

    pcm_data = audio_i2f(read_audio_file('ParaJet/examples/test.wav'))
    pcm_data = reshape_audio_to_BxT(pcm_data, T)

    # 构造固定输入
    N = pcm_data.shape[0]
    if N < B:
        padding = np.zeros((B - N, T), dtype=pcm_data.dtype)
        pcm_data = np.concatenate([pcm_data, padding], axis=0)
    elif N > B:
        pcm_data = pcm_data[:B]

    hotwords = ["新声招聘" for _ in range(len(pcm_data))]
    print(model(pcm_data, hotwords))
