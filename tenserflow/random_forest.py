import tflearn
from tflearn.estimators import RandomForestClassifier
from tflearn.data_utils import load_csv
import numpy as np
from datetime import datetime


data, labels = load_csv('train.txt', target_column=13, categorical_labels=True, n_classes=2)

# Transformation lists
# US size map
size_labels = ['?', 'unsized', 'xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxxl']
size_values = [0, 0, 30, 32, 35, 41, 43, 46, 48, 50]
gender_labels = ['?', 'Mrs', 'Mr', 'Family', 'not reported', 'Company']
gender_values = [0, 2, 3, 4, 1, 5]
color_labels = [
    '?', 'denim', 'ocher', 'curry', 'green', 'black', 'brown', 'red', 'mocca', 'anthracite', 'olive',
    'petrol', 'blue', 'grey', 'beige', 'ecru', 'turquoise', 'magenta', 'purple', 'pink', 'khaki',
    'navy', 'habana', 'silver', 'white', 'nature', 'stained', 'orange', 'azure', 'apricot', 'mango',
    'berry', 'ash', 'hibiscus', 'fuchsia', 'blau', 'dark denim', 'mint', 'ivory', 'yellow', 'bordeaux',
    'pallid', 'ancient', 'baltic blue', 'almond', 'aquamarine', 'brwon', 'aubergine', 'aqua', 'dark garnet',
    'dark grey', 'avocado', 'creme', 'champagner', 'cortina mocca', 'currant purple', 'cognac',
    'aviator', 'gold', 'ebony', 'cobalt blue', 'kanel', 'curled', 'caramel', 'antique pink', 'darkblue',
    'copper coin', 'terracotta', 'basalt', 'amethyst', 'coral', 'jade', 'opal', 'striped', 'mahagoni',
    'floral', 'dark navy', 'dark oliv', 'vanille', 'ingwer', 'iron', 'graphite', 'leopard', 'oliv', 'bronze',
    'crimson', 'lemon', 'perlmutt'
]
color_values = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    84, 85, 86, 87
]
state_labels = [
    '?', 'Baden-Wuerttemberg', 'Saxony', 'Rhineland-Palatinate', 'North Rhine-Westphalia', 'Berlin', 'Bavaria',
    'Hesse', 'Bremen', 'Schleswig-Holstein', 'Brandenburg', 'Hamburg', 'Lower Saxony', 'Saxony-Anhalt',
    'Saarland', 'Thuringia', 'Mecklenburg-Western Pomerania'
]
state_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def prepare_data(X, ignored_columns, offset=0):
    now = datetime.now()
    for i in range(len(X)):
        order_date = datetime.strptime(X[i][1-offset], '%Y-%m-%d')
        X[i][1-offset] = (now - datetime.strptime(X[i][1-offset], '%Y-%m-%d')).days
        X[i][12-offset] = (now - datetime.strptime(X[i][12-offset], '%Y-%m-%d')).days
        # Delivery dates
        delivery_date = X[i][2-offset]
        if delivery_date != '?':
            delivery_date = datetime.strptime(delivery_date, '%Y-%m-%d')
            X[i][2-offset] = (delivery_date - order_date).days
            if X[i][2-offset] < 0:
                X[i][2-offset] = None
        else:
            X[i][2-offset] = None
        # Ages
        birth_date = X[i][10-offset]
        if birth_date != '?':
            birth_date = datetime.strptime(birth_date, '%Y-%m-%d')
            X[i][10-offset] = (now - birth_date).days
        else:
            X[i][10-offset] = None
        # transform size
        size = X[i][4-offset].lower()
        if '+' in size:
            X[i][4-offset] = int(size.replace('+', '')) + 1
        elif size in size_labels:
            X[i][4-offset] = size_values[size_labels.index(size)]
        # transform color
        color = X[i][5-offset]
        if color in color_labels:
            X[i][5-offset] = color_values[color_labels.index(color)]
        # transform salutation
        salutation = X[i][9-offset]
        if salutation in gender_labels:
            X[i][9-offset] = gender_values[gender_labels.index(salutation)]
        # transform state
        state = X[i][11-offset]
        if state in state_labels:
            X[i][11-offset] = state_values[state_labels.index(state)]
    # transform ages and delivery dates to mean values
    birth_dates = filter(None, np.array(X)[:, 10-offset])
    mean_birthday = int(np.array(list(birth_dates)).mean())
    delivery_dates = filter(None, np.array(X)[:, 2-offset])
    mean_delivery = int(np.array(list(delivery_dates)).mean())
    for i in range(len(X)):
        if not X[i][10-offset]:
            X[i][10-offset] = mean_birthday
        if not X[i][2-offset]:
            X[i][2-offset] = mean_delivery
    
    X = np.array(X, dtype=np.float32)
    return np.delete(X, ignored_columns, axis=1)


data = prepare_data(data, [0])
labels = np.array(labels, dtype=np.int32)

model = RandomForestClassifier(n_classes=2, n_estimators=10, max_nodes=1000)
model.fit(data, labels[:, 1], batch_size=10000, display_step=10)

print("Accuracy:")
print(model.evaluate(data, labels[:, 1], tflearn.accuracy_op))

data, labels = load_csv('test.txt', target_column=0)
data = prepare_data(data, [], offset=1)

values = model.predict(data)

with open('result.txt', 'w') as out_file:
    out_file.write('orderItemID,returnShipment\n')
    for i in range(len(values)):
        out_file.write("{item},{prediction}\n".format(
            item=labels[i],
            prediction=values[i]
        ))
