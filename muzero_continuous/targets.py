"""Цель ценности для непрерывной задачи с внешними ограничениями времени."""
import math


def value_target(rewards, terminated, truncated, bootstrap_value, gamma):
    """Возврат по одному фрагменту; после внешнего обрыва оценка сохраняется.

    Если фрагмент короче желаемого горизонта, вызывающий код передаёт
    bootstrap_value последнего реально достигнутого наблюдения.
    Фрагмент не должен пересекать границу эпизода.
    """
    if not (len(rewards) == len(terminated) == len(truncated)) or not rewards:
        raise ValueError("Нужен непустой фрагмент согласованной длины")
    if not 0 <= gamma <= 1 or not math.isfinite(bootstrap_value):
        raise ValueError("Неверное дисконтирование или оценка")
    if any(not math.isfinite(float(r)) for r in rewards):
        raise ValueError("Награды должны быть конечными")
    if any(t or u for t,u in zip(terminated[:-1],truncated[:-1])):
        raise ValueError("Фрагмент пересекает конец эпизода")
    tail = 0.0 if terminated[-1] else float(bootstrap_value)
    for reward in reversed(rewards):
        tail = float(reward) + gamma*tail
    return tail
