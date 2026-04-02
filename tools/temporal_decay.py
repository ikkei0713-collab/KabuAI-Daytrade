"""
Temporal Decay Utility

Applies exponential time decay to trade pattern scores.
Based on the Dexter financial research agent's approach to
weighting historical knowledge by recency.
"""

import math


def apply_decay(score: float, age_days: float, half_life: float = 10.0) -> float:
    """Apply exponential time decay to a score.

    Args:
        score: Original score value.
        age_days: Age of the observation in trading days.
        half_life: Number of trading days for the score to halve (default 10).

    Returns:
        Decayed score.  After *half_life* trading days the returned value
        is approximately ``score / 2``.
    """
    if half_life <= 0:
        return score
    decay_rate = math.log(2) / half_life
    return score * math.exp(-decay_rate * age_days)
