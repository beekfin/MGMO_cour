"""Проверяемое преобразование сетки в численное действие среды."""
import numpy as np


def uniform_grid(low: float, high: float, count: int = 5) -> np.ndarray:
    if not np.isfinite([low, high]).all() or low >= high or count < 2:
        raise ValueError("Нужны конечные low < high и не меньше двух узлов")
    return np.linspace(low, high, count, dtype=np.float32)


def environment_action(grid: np.ndarray, index: int) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float32)
    if grid.ndim != 1 or not np.isfinite(grid).all():
        raise ValueError("Сетка должна быть конечным одномерным массивом")
    if not isinstance(index, (int, np.integer)) or not 0 <= index < len(grid):
        raise ValueError("Индекс выходит за границы сетки")
    return np.array([grid[index]], dtype=np.float32)
