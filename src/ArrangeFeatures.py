import numpy as np
from tabulate import tabulate
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def risker(u, risk_dict):
    try : 
        return risk_dict[u]
    except: 
        return 0.5

# Calculates New features from data combines and returns them.
def get_new_features(data, risk_dict_itemID, risk_dict_customerID, risk_dict_manufacturerID
                    , cust_prob_dict, purchasePerUserID, purchaseMean
                    , averagePricePerID, averagePrice ):
    # How long each delivery took
    transmit_time = np.asmatrix(find_delivery_duration(data))
    # Customer age list in years
    age_list = np.asmatrix(get_customer_age(data))
    # Customer account age in days
    account_age_list = np.asmatrix(customer_account_age(data))
    # Distance from new year
    newyear_distance = np.asmatrix(order_newyear_date_distance(data))
    # Distance from lovers day
    lovers_distance = np.asmatrix(order_lovers_date_distance(data))
    # Risk values from itemID
    risk_feature_itemID = np.asmatrix([risker(u, risk_dict_itemID) for u in data['itemID']])
    risk_feature_customerID = np.asmatrix([risker(u, risk_dict_customerID) for u in data['customerID']])
    risk_feature_manufacturerID = np.asmatrix([risker(u, risk_dict_manufacturerID) for u in data['manufacturerID']])
    # purchase_count_per_user
    purchase_count_per_user = []
    for d in data['customerID']:
        try:
            purchase_count_per_user.append(purchasePerUserID[d])
        except:
            purchase_count_per_user.append(purchaseMean)
    purchase_count_per_user = np.asmatrix(purchase_count_per_user)
    # Average price paid per user
    average_paid_per_user = []
    for d in data['customerID']:
        try:
            average_paid_per_user.append(averagePricePerID[d])
        except:
            average_paid_per_user.append(averagePrice)
    average_paid_per_user = np.asmatrix(average_paid_per_user)
    # Probability from all orders for each customer
    customer_probability = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        try:
            customer_probability[i] = cust_prob_dict[data['customerID'][i]]
        except:
            customer_probability[i] = 0.5
    customer_probability = np.asmatrix(customer_probability)
    # Concatenate the data
    features = np.concatenate((customer_probability.T, average_paid_per_user.T, purchase_count_per_user.T,
                            risk_feature_manufacturerID.T, risk_feature_customerID.T, risk_feature_itemID.T,
                            lovers_distance.T, newyear_distance.T, transmit_time.T, age_list.T,
                            account_age_list.T), axis=1)

    return features

# How long each delivery took
def find_delivery_duration(data):
    delivery_duration = []
    for order,deliv in zip(data['orderDate'], data['deliveryDate']):
        order = datetime.strptime(order, '%Y-%m-%d')
        deliv = datetime.strptime(deliv, '%Y-%m-%d')
        delivery_duration.append( (deliv - order).days )
    return delivery_duration


# Customer age list in years
def get_customer_age(data):
    customer_age = []
    for start, end in zip(data['orderDate'], data['dateOfBirth']):
        start = datetime.strptime(start, '%Y-%m-%d')
        end = datetime.strptime(end, '%Y-%m-%d')
        customer_age.append( relativedelta(start, end).years )
    return customer_age

# Customer account age in days
def customer_account_age(data):
    account_age = []
    for start, end in zip(data['orderDate'], data['creationDate']):
        start = datetime.strptime(start, '%Y-%m-%d')
        end = datetime.strptime(end, '%Y-%m-%d')
        account_age.append( relativedelta(start, end).days )
    return account_age

# How further is orderDate from new year?
def order_newyear_date_distance(data):
    return [ (datetime.strptime(x, '%Y-%m-%d').month == 1 or datetime.strptime(x, '%Y-%m-%d').month == 12 ) for x in data['orderDate']  ]
# (datetime.strptime(x, '%Y-%m-%d').month() == 1 or datetime.strptime(x, '%Y-%m-%d').month() == 12 )
# How further is orderDate from lovers date?
def order_lovers_date_distance(data):
    return [ (
            datetime.strptime(x, '%Y-%m-%d').month == 2 or 
            (datetime.strptime(x, '%Y-%m-%d').month == 3 and datetime.strptime(x, '%Y-%m-%d').day < 15) or
            (datetime.strptime(x, '%Y-%m-%d').month == 1 and datetime.strptime(x, '%Y-%m-%d').day > 15)
            ) 
            for x in data['orderDate']  ]

# ItemID, Manufacturer ID, and Cusomer ID risks
# Calculate risk of an item being returned
def calculate_return_risk(data, header):
    # Get all unique ids
    unique, counts = np.unique(data[header], return_counts=True)
    error_dict = {}
    success_dict = {}
    risk_dict = {}

    for u in unique:
        error_dict[u] = 0
        success_dict[u] = 0
        risk_dict[u] = 0

    for (itemid, returnVal) in zip(data[header], data['returnShipment']): 
        if returnVal == 1: 
            error_dict[itemid] += 1 
        else: 
            success_dict[itemid] += 1

    # Calculate Risk
    c = 5
    for u in unique:
        risk_dict[u] = 1.0 - np.divide( success_dict[u] ** c + 500.0 , (success_dict[u] + error_dict[u]) ** c + 1000.0)
    
    return risk_dict
