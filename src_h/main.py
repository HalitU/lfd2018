"""
Main.py program flow:
get data => Fix missing values => Get new features => Get original Data
features => Fit to model => Predict
"""

from PrintStats import *
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm, cross_validation
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor , RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# import tflearn
# import tensorflow as tf

import time
start_time = time.clock()

# ---------------- TRAINING PART -----------------------------
train_data = pd.read_csv('train.txt', sep=',')

# Fix order and delivery date data
order_deliver_average_fix = calculate_order_delivery_average(train_data['orderDate'], train_data['deliveryDate'])

train_data['deliveryDate'] = fix_order_delivery_data(train_data['orderDate'], train_data['deliveryDate'], order_deliver_average_fix)

# Fill missing color data
fix_color(train_data['color'])

# Get purchase count for users
purchasePerUserID = train_data.groupby('customerID').count().to_dict()
purchaseMean = train_data.groupby('customerID').count()['orderDate'].mean()

# Get average price for users
averagePricePerID = train_data.groupby('customerID')['price'].mean().to_dict()
averagePrice = train_data.groupby('customerID')['price'].mean().sum() / train_data.groupby('customerID')['price'].mean().count()

# Remove unnecessary date of birth data
unique_customerID_indices = train_data['customerID'].drop_duplicates().index
birth_average_fix = calculate_birth_average(train_data['dateOfBirth'][unique_customerID_indices])

fix_birth_with_average(train_data['dateOfBirth'], birth_average_fix)

# Customer Probability Dict
ship_sums = train_data.groupby(['customerID']).sum()['returnShipment']
ship_counts = train_data.groupby(['customerID']).count()['returnShipment']
cust_prob_dict = (ship_sums / ship_counts).to_dict()

# Risk Calculations
risk_dict_itemID = calculate_return_risk(train_data, 'itemID')
risk_dict_customerID = calculate_return_risk(train_data, 'customerID')
risk_dict_manufacturerID = calculate_return_risk(train_data, 'manufacturerID')

# Get new features
train_features = get_new_features(train_data, risk_dict_itemID, risk_dict_customerID,
                                 risk_dict_manufacturerID, cust_prob_dict, purchasePerUserID, purchaseMean
                                 , averagePricePerID, averagePrice )

# Change strings with integer mapping for using more features from original data
color = get_data_mapping(train_data['color'])
size = get_data_mapping(train_data['size'])
salutation = get_data_mapping(train_data['salutation'])
state = get_data_mapping(train_data['state'])

train_data = train_data.replace({'color': color, 'size': size, 'salutation': salutation, 'state': state })

# Connect all features together
train_features = pd.DataFrame(train_features)
train_features = train_features.assign(size=train_data['size'], color=train_data['color'], price=train_data['price'],
                                        salutation=train_data['salutation'], state=train_data['state'])

# Confirming the size of the data
print("Label size: ", len(train_data['returnShipment']))
print("Feature size: ", train_features.shape)

# -------------- TEST SET PART -------------------------------------
test_data = pd.read_csv('test.txt', sep=',')
# Fix test data
# We have already calculated the average values from training set
print("-------Working on Test Set---------------")
test_data['deliveryDate'] = fix_order_delivery_data(test_data['orderDate'], test_data['deliveryDate'], order_deliver_average_fix)
fix_color(test_data['color'])
fix_birth_with_average(test_data['dateOfBirth'], birth_average_fix)
print_data_stats(test_data)

# Change color strings with indeces for putting into model
test_data = test_data.replace({'color': color, 'size': size, 'salutation': salutation, 'state': state })

# Create New Features from test set
test_features = pd.DataFrame(get_new_features(test_data, risk_dict_itemID, risk_dict_customerID
                        , risk_dict_manufacturerID, cust_prob_dict, purchasePerUserID, purchaseMean
                        , averagePricePerID, averagePrice))

test_features = test_features.assign(size=test_data['size'], color=test_data['color'], price=test_data['price'],
                                    salutation=test_data['salutation'], state=test_data['state'])

print("Test feature size: ", test_features.shape)

# @NOTE Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, max_features=16, criterion='entropy', n_jobs=-1)
rfc.fit(train_features, train_data['returnShipment'])
predictions = rfc.predict(test_features)

# @NOTE DNN Tenserflow
# tf.reset_default_graph()
# net = tflearn.input_data(shape=[None, 16])
# net = tflearn.fully_connected(net, 100)
# net = tflearn.fully_connected(net, 50)
# net = tflearn.fully_connected(net, 2, activation='softmax')
# net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
#
# train_features = np.array(train_features, dtype=np.float32)
# train_labels = np.array(train_data['returnShipment'], dtype=np.int32)
# train_labels = train_labels.reshape(train_labels.shape[0], 1)
# train_labels = np.concatenate((1-train_labels, train_labels), axis=1)
#
# model = tflearn.DNN(net)
# model.fit(train_features, train_labels, n_epoch=20, batch_size=500, show_metric=True, validation_set=0.1)
# predictions = model.predict(test_features)

# @NOTE K Neighborhood Classifier
# knc = KNeighborsClassifier()
# knc.fit(train_features, train_data['returnShipment'])
# predictions = knc.predict(test_features)

# @NOTE ExtraTreeRegressor
# etr = ExtraTreesRegressor()
# etr.fit(train_features, train_data['returnShipment'])
# predictions = etr.predict(test_features)

# @TODO SVM Classification with PCA
# X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(train_features, train_data['returnShipment'], test_size=0.5)
# print(X_train.shape)
# pca = PCA(n_components=2)
# pca.fit(X_train)
# X_pca_train = pca.transform(X_train)
# X_pca_test = pca.transform(X_test)
# print(X_pca_train)
# sv = svm.SVC()
# sv.fit(X_pca_train, Y_train)
# accuracy = sv.score(X_pca_test, Y_test)
# print("Accuracy:")
# print(accuracy)
# test_pca_features = pca.transform(test_features)
# predictions = sv.predict(test_pca_features)

# -----------------Prediction Done----------------------------
print(predictions.shape)
print(np.asmatrix(predictions).shape)
# predictions = predictions[:, 1]
predictions = np.asmatrix(predictions).T
predictions = pd.DataFrame(predictions)
predictions.index += 1
predictions.index.name = 'orderItemID'
predictions.to_csv("predictions.csv", header=['returnShipment'], float_format="%.10f", index_label='orderItemID')
print("Prediction Done.")

# Bagging, SVM, Kmeans, Kneigh, ExtraTreesRegressor
# naive_bayes : multinomialNB, bernoulliNB
# linear_model : Perceptron == SGDClassifier, PassiveAggressiveClassifier
# neural_network: MLPClassifier

# @NOTE Native bagging 3514 best
# Bagging with KMeans invalid results
# Bagging with ExtraTreeRegressor memory error
# Bagging Regressor
# bag = BaggingRegressor(ExtraTreesRegressor())
# bag.fit(train_features, train_data['returnShipment'])
# predictions = bag.predict(test_features)

# @NOTE Native KMeans invalid results
# km = KMeans()
# km.fit(train_features, train_data['returnShipment'])
# predictions = km.predict(test_features)

# SVM Regressor Didnt work
# sv = svm.SVR()
# sv.fit(train_features, train_data['returnShipment'])
# predictions = sv.predict(test_features)

# MultinomialNB 11k
# mNB = MultinomialNB()
# mNB.fit(train_features, train_data['returnShipment'])
# predictions = mNB.predict(test_features)

# SGD Classifier 10.6k
# sgd = SGDClassifier()
# sgd.fit(train_features, train_data['returnShipment'])
# predictions = sgd.predict(test_features)

# SGD Regressor Values too HIGH
# sgd = SGDRegressor()
# sgd.fit(train_features, train_data['returnShipment'])
# predictions = sgd.predict(test_features)

# PassiveAggressiveRegressor NOPE negative values
# par = PassiveAggressiveRegressor()
# par.fit(train_features, train_data['returnShipment'])
# predictions = par.predict(test_features)

# PassiveAggressiveClassifier 8.6k
# clf = PassiveAggressiveClassifier()
# clf.fit(train_features, train_data['returnShipment'])
# predictions = clf.predict(test_features)

# Random Forest Regressor 3.5k
# rf = RandomForestRegressor(n_estimators=100, random_state=10, n_jobs=-1)
# rf.fit(train_features, train_data['returnShipment'])
# predictions = rf.predict(test_features)

# MLPClassifier 7k
# nn = MLPClassifier()
# nn.fit(train_features, train_data['returnShipment'])
# predictions = nn.predict(test_features)

# MLPRegressor NOPE negative values
# mlp = MLPRegressor()
# mlp.fit(train_features, train_data['returnShipment'])
# predictions = mlp.predict(test_features)
