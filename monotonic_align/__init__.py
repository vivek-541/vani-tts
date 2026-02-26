import numpy as np
import torch

def maximum_path(value, mask):
    """Monotonic alignment search - pure numpy implementation"""
    value = value.cpu().numpy().astype(np.float32)
    mask = mask.cpu().numpy().astype(np.int32)
    b, x_max, y_max = value.shape
    paths = np.zeros_like(value, dtype=np.int32)
    
    for i in range(b):
        t_x = mask[i, :, 0].sum()
        t_y = mask[i, 0, :].sum()
        v = value[i, :t_x, :t_y]
        path = np.zeros_like(v, dtype=np.int32)
        
        dp = np.zeros_like(v)
        dp[0, 0] = v[0, 0]
        for j in range(1, t_y):
            dp[0, j] = -1e9
        for j in range(1, t_x):
            dp[j, 0] = dp[j-1, 0] + v[j, 0]
        for j in range(1, t_x):
            for k in range(1, t_y):
                dp[j, k] = max(dp[j-1, k], dp[j-1, k-1]) + v[j, k]
        
        # Backtrack
        j, k = t_x - 1, t_y - 1
        while j >= 0 and k >= 0:
            path[j, k] = 1
            if k == 0 or (j > 0 and dp[j-1, k] > dp[j-1, k-1]):
                j -= 1
            else:
                j -= 1
                k -= 1
        
        paths[i, :t_x, :t_y] = path
    
    return torch.from_numpy(paths).to(torch.float32)

def mask_from_lens(lens, max_len=None, step=1):
    # When called as mask_from_lens(attn, input_lengths, mel_lengths)
    # step may be a per-sample tensor of lengths
    if isinstance(step, torch.Tensor) and step.numel() > 1:
        # step is actually per-sample mel lengths
        lengths = step
        if isinstance(max_len, torch.Tensor):
            max_len_val = int(max_len.max().item())
        elif max_len is not None:
            max_len_val = int(max_len)
        else:
            max_len_val = int(lengths.max().item())
        ids = torch.arange(0, max_len_val, device=lengths.device)
        return ids.unsqueeze(0) < lengths.unsqueeze(1)
    # Normal case: lens is lengths, step is scalar
    if isinstance(max_len, torch.Tensor):
        max_len = int(max_len.max().item())
    if max_len is None:
        max_len = int(lens.max().item())
    step = int(step.item()) if isinstance(step, torch.Tensor) else int(step)
    if step > 1:
        max_len = max_len // step
        lens = lens // step
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    return ids < lens.unsqueeze(1)
