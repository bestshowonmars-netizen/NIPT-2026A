from numba import njit

@njit(cache=True)
def surface_conductance(coefficient, exchange, distance, area):
    if exchange == 0.0:
        return 0.0
    return area / (distance / coefficient + 1.0 / exchange)

@njit(cache=True)
def surface_value(cell_value, environment, coefficient, exchange, distance):
    return cell_value + exchange * distance / (coefficient + exchange * distance) * (environment - cell_value)
