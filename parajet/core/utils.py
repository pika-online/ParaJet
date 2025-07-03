from parajet.utils import *

def cif(hidden, alphas, threshold=1.0,max_len=100):
    B, F, C = hidden.shape

    # Padding: Add dummy frame and alpha
    hidden = np.concatenate((hidden, np.zeros((B, 1, C), dtype='float32')), axis=1)
    F += 1
    pad_alpha = np.full((B, 1), 0.45, dtype='float32')
    alphas = np.concatenate((alphas.squeeze(-1), pad_alpha), axis=1)

    integrate = np.zeros((B,), dtype='float32')
    frame = np.zeros((B, C), dtype='float32')

    final_frames = [[] for _ in range(B)]

    for t in range(F):
        alpha_t = alphas[:, t]
        distribution_completion = threshold - integrate
        integrate += alpha_t

        flags = integrate >= threshold
        cur = np.where(flags, distribution_completion, alpha_t)
        remains = alpha_t - cur

        # Add weighted contribution to frame
        frame += (cur[:, None] * hidden[:, t, :])

        for b in range(B):
            if flags[b]:
                final_frames[b].append(frame[b].copy())
                frame[b] = remains[b] * hidden[b, t, :]

        integrate = np.where(flags, integrate - threshold, integrate)

    # Finalize output
    final_frame_list = []
    max_label_lens = []

    for b in range(B):
        label_len = int(np.round(alphas[b].sum()))
        max_label_lens.append(label_len)
        frames = final_frames[b]
        if label_len > len(frames):
            frames.append(np.zeros((C,), dtype='float32'))
        final_frame_list.append(np.stack(frames, axis=0))

    # Pad to the same length
    padded_frames = np.zeros((B, max_len, C), dtype='float32')
    for b in range(B):
        length = final_frame_list[b].shape[0]
        padded_frames[b, :length, :] = final_frame_list[b]

    return padded_frames, max_label_lens



def cif2(tokens:list,alphas,threshold=1.0,audio_seconds=30):

    intergrate = 0.
    ans = []
    num = len(alphas)
    for i in range(len(alphas)):
        intergrate += alphas[i]
        if intergrate>=threshold:
            ans.append((tokens.pop(0),round(i/num*audio_seconds,3)))
            intergrate -= threshold
    return ans

def maxSumSubarrayWithGaps(nums,k,gap):
    nums_ = list(nums.copy())
    nums_.insert(0,0)
    N = len(nums_)

    # 初始化表单
    dp = [[0 for j in range(k+1)] for _ in range(N)]
    path = [[[] for j in range(k+1)] for _ in range(N)]
    # 初始化边界
    for i in range(N): # dp[:,0]
        dp[i][0] = 0
        path[i][0] = []
    for j in range(k+1): # dp[0,:]
        dp[0][j] = 0
        path[0][j] = []
    # 定义t = 1
    for j in range(k+1): # dp[1,:]
        if j==0:
            dp[1][j] = 0
            path[1][j] = []
        elif j==1:
            dp[1][j] = nums_[j]
            path[1][j] = []
        else:
            dp[1][j] = 0
            path[1][j] = []

    for j in range(1,k+1):
        for i in range(2,N):
            context = (j-1)*gap+1 # 保证content所需帧数
            if i>=context:
                if dp[i-1][j]>=dp[i-gap][j-1]+nums_[i]:
                    dp[i][j] = dp[i-1][j]
                    path[i][j] = path[i-1][j]
                else:
                    dp[i][j] = dp[i-gap][j-1]+nums_[i]
                    path[i][j] = [(i-gap,j-1),(i,j)]

    # for i,p in enumerate(dp):
    #     print(f"{i}|",p)
    # print()
    # for i,p in enumerate(path):
    #     print(f"{i}|",p)
        
    # 回溯
    max_score = dp[-1][-1]
    max_path = []
    a,b = -1,-1
    while path[a][b]:
        link,res = path[a][b]
        max_path.append(res)
        a,b = link
    max_path.reverse()
    return max_score,max_path

