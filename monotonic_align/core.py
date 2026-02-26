import numpy as np

def maximum_path_c(paths, values, t_xs, t_ys):
    """C-compatible maximum path function"""
    b = values.shape[0]
    for i in range(b):
        t_x = t_xs[i]
        t_y = t_ys[i]
        v = values[i, :t_x, :t_y]
        dp = np.full_like(v, -1e9)
        dp[0, 0] = v[0, 0]
        for j in range(1, t_x):
            dp[j, 0] = dp[j-1, 0] + v[j, 0]
        for j in range(1, t_x):
            for k in range(1, t_y):
                dp[j, k] = max(dp[j-1, k], dp[j-1, k-1]) + v[j, k]
        j, k = t_x - 1, t_y - 1
        while j >= 0 and k >= 0:
            paths[i, j, k] = 1
            if k == 0 or (j > 0 and dp[j-1, k] >= dp[j-1, k-1]):
                j -= 1
            else:
                j -= 1
                k -= 1
