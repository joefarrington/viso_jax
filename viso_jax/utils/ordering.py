import math
from scipy.stats import poisson


def calculate_newsvendor_order_quantity_poisson_demand(
    demand_poisson_mean: float,
    max_useful_life: float,
    sales_price: float,
    variable_order_cost: float,
) -> int:
    """Calculate newsvendor order quantity for a  perishable product with poisson demand based on
    Equation 2 of Hendrix et al (2019)"""
    quantile = (sales_price - variable_order_cost) / sales_price
    return math.ceil(poisson.ppf(quantile, max_useful_life * demand_poisson_mean))
