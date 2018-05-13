import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


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

X = []
Y = []

with open('train.txt', 'r') as infile:
    for line in infile.readlines():
        line = line.replace('\n', '')
        X.append(line.split(','))
X = X[1:]
# Convert data
now = datetime.now()
for i in range(len(X)):
    X[i][1] = (now - datetime.strptime(X[i][1], '%Y-%m-%d')).days
    X[i][12] = (now - datetime.strptime(X[i][12], '%Y-%m-%d')).days
    # Delivery dates
    delivery_date = X[i][2]
    if delivery_date != '?':
        delivery_date = datetime.strptime(delivery_date, '%Y-%m-%d')
        X[i][2] = (now - delivery_date).days
    else:
        X[i][2] = None
    # Ages
    birth_date = X[i][10]
    if birth_date != '?':
        birth_date = datetime.strptime(birth_date, '%Y-%m-%d')
        X[i][10] = (now - birth_date).days
    else:
        X[i][10] = None
    # transform size
    size = X[i][4].lower()
    if '+' in size:
        X[i][4] = int(size.replace('+', '')) + 1
    elif size in size_labels:
        X[i][4] = size_values[size_labels.index(size)]
    # transform color
    color = X[i][5]
    if color in color_labels:
        X[i][5] = color_values[color_labels.index(color)]
    # transform salutation
    salutation = X[i][9]
    if salutation in gender_labels:
        X[i][9] = gender_values[gender_labels.index(salutation)]
    # transform state
    state = X[i][11]
    if state in state_labels:
        X[i][11] = state_values[state_labels.index(state)]

X = np.array(X, dtype=np.float64)
Y = X[:, 13]

# Delivery data
delivery_x = X[:, 1] - X[:, 2]
plt.scatter(delivery_x[10000:20000], Y[10000:20000])
plt.savefig('graph/delivery.png')
plt.close()

plt.scatter(X[10000:20000, 3], Y[10000:20000])
plt.savefig('graph/itemID.png')
plt.close()

plt.scatter(X[10000:20000, 4], Y[10000:20000])
plt.savefig('graph/size.png')
plt.close()

plt.scatter(X[10000:20000, 5], Y[10000:20000])
plt.savefig('graph/color.png')
plt.close()

plt.scatter(X[10000:20000, 6], Y[10000:20000])
plt.savefig('graph/manufacturerID.png')
plt.close()

plt.scatter(X[10000:20000, 7], Y[10000:20000])
plt.savefig('graph/price.png')
plt.close()

plt.scatter(X[10000:20000, 8], Y[10000:20000])
plt.savefig('graph/customerID.png')
plt.close()

plt.scatter(X[10000:20000, 9], Y[10000:20000])
plt.savefig('graph/salutation.png')
plt.close()

plt.scatter(X[10000:20000, 10], Y[10000:20000])
plt.savefig('graph/dateOfBirth.png')
plt.close()

plt.scatter(X[10000:20000, 11], Y[10000:20000])
plt.savefig('graph/state.png')
plt.close()

plt.scatter(X[10000:20000, 12], Y[10000:20000])
plt.savefig('graph/creationDate.png')
plt.close()
