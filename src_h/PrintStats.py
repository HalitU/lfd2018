import numpy as np
from tabulate import tabulate
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestRegressor

from CleanData import *
from ArrangeFeatures import *

# Prints the current missing data count for eacher header in a nicer way.
def print_data_stats(data):
    table_data = []
    for d in data.columns:
        try:
            table_data.append([d, data[d].value_counts()['?'] ])
        except:
            table_data.append([d, 0])
    print(tabulate(table_data, headers=['Feature Name', 'Missing Count']))































