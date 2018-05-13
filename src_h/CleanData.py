import numpy as np
from datetime import datetime

# Calculates the average time between each order and delivery
# while discarding '?' missing character and invalid values
# such as orderDate > deliveryDate
def calculate_order_delivery_average(orderData, deliveryData):
    data_count = 0
    date_sum = None
    error_date_count = 0

    for (order, deliv) in zip(orderData, deliveryData):
        if deliv != '?':
            deliv = datetime.strptime(deliv, '%Y-%m-%d')
            # If order date is smaller than the smallest order date there must be an error!
            if deliv > datetime.strptime('2012-01-01', '%Y-%m-%d'):
                order = datetime.strptime(order, '%Y-%m-%d')
                # print(deliv - order)
                if not date_sum:
                    date_sum = deliv - order
                else:
                    date_sum += (deliv-order)
                data_count += 1
            else:
                error_date_count += 1

    return date_sum / data_count

# Replaces invalid values with the average date
def fix_order_delivery_data(orderData, deliveryData, average_date):
    minimum_date = datetime.strptime('2012-01-01', '%Y-%m-%d')
    index = 0
    new_array = []
    for o, d in zip(orderData, deliveryData):
        index += 1
        if d == '?':
            new_array.append(str(( datetime.strptime(o, '%Y-%m-%d') + average_date ).date()))
        elif datetime.strptime(d, '%Y-%m-%d') < minimum_date:
            new_array.append(str((datetime.strptime(o, '%Y-%m-%d') + average_date).date()))
        else:
            new_array.append(d)
    return np.asarray(new_array)

# Fixing missing colors with the most common color in the set
def fix_color(colorData):
    unique, counts = np.unique(colorData, return_counts=True)
    ind = np.argmax(counts)
    max_color = unique[ind]
    colorData[colorData == '?'] = max_color

# Calculates the average time between a start date and valid 
# birth dates while discarding '?' missing character and 
# values such as birthDate < 1910.01.01
def calculate_birth_average(birthDateData):
    data_count = 0
    date_sum = None
    error_date_count = 0
    minimum_date = datetime.strptime('1910-01-01', '%Y-%m-%d')

    start = datetime.strptime('1700-01-01', '%Y-%m-%d')
    for birth in birthDateData:
        if birth != '?':
            birth = datetime.strptime(birth, '%Y-%m-%d')
            # If order date is smaller than the smallest order date there must be an error!
            if birth > minimum_date:
                if not date_sum:
                    date_sum = (birth - start) / 100000
                else:
                    date_sum += (birth - start) / 100000
                data_count += 1
            else:
                error_date_count += 1

    average_date = (date_sum / data_count) * 100000
    average_date_f = str((start + average_date).date())
    return average_date_f

# Replaces invalid values with the average birth date
def fix_birth_with_average(birthDateData, average_date_f):
    minimum_date = datetime.strptime('1910-01-01', '%Y-%m-%d')
    birthDateData[birthDateData == '?'] = average_date_f
    birthDateData[birthDateData.apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) < minimum_date] = average_date_f

# Gets the mapping of unique non invalid elements for converting
# non-float numbers into integers for taking them as features.
def get_data_mapping(data):
    unique, count = np.unique(data, return_counts=True)
    unique = np.delete(unique, np.where(unique == '?'))
    color_mapping = {}
    index = 0
    for c in unique:
        color_mapping[c] = index
        index += 1
    return color_mapping
