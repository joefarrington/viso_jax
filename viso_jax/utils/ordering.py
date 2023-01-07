# Calculate max order quantity for perishable product using newsvendor method
# Assumes demand is poisson distributed

import math
from scipy.stats import poisson


def calculate_newsvendor_order_quantity_poisson_demand(
    demand_poisson_mean, max_useful_life, sales_price, variable_order_cost
):
    quantile = (sales_price - variable_order_cost) / sales_price
    return math.ceil(poisson.ppf(quantile, max_useful_life * demand_poisson_mean))


def hand_coded_order_quantity(order_quantity):
    return order_quantity
