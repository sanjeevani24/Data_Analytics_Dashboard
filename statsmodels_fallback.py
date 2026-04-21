"""Fallback simple forecast when statsmodels is unavailable."""
import numpy as np


def simple_forecast(data: np.ndarray, steps: int):
    """Simple exponential smoothing with additive seasonality (period=12)."""
    period = 12
    alpha = 0.3  # level smoothing
    gamma = 0.1  # seasonal smoothing

    n = len(data)
    level = np.zeros(n)
    seasonal = np.zeros(n + period)

    # Init
    level[0] = np.mean(data[:period]) if n >= period else data[0]
    for i in range(period):
        seasonal[i] = data[i] / level[0] if level[0] != 0 else 1.0

    for t in range(1, n):
        s_idx = t % period
        prev_level = level[t - 1]
        prev_seasonal = seasonal[t - 1 + period - (t % period) if t < period else s_idx]
        # Update level
        level[t] = alpha * (data[t] / (seasonal[s_idx] or 1)) + (1 - alpha) * prev_level
        # Update seasonal
        seasonal[t + period] = gamma * (data[t] / (level[t] or 1)) + (1 - gamma) * seasonal[s_idx]

    # Forecast
    forecasts = []
    last_level = level[-1]
    for h in range(1, steps + 1):
        s_idx = (n + h - 1) % period
        forecasts.append(last_level * (seasonal[n + s_idx] or 1))

    return np.array(forecasts), None
