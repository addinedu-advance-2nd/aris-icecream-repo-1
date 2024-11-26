import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
import sqlite3

# Define the menu and topping options
menu = {
    'flavors': ['Banana', 'Chocolate', 'Strawberry'],
    'toppings': ['Lotus', 'Rainbow', 'Oreo']
}

# Example training data (each menu and topping, and sales)
menu_data = [
    ('Banana', 'Lotus', 150),
    ('Chocolate', 'Rainbow', 200),
    ('Strawberry', 'Oreo', 180),
    ('Banana', 'Rainbow', 130),
    ('Chocolate', 'Lotus', 210),
    ('Strawberry', 'Lotus', 170)
]

# Convert flavors and toppings to index values (for one-hot encoding)
flavor_map = {flavor: idx for idx, flavor in enumerate(menu['flavors'])}
topping_map = {topping: idx for idx, topping in enumerate(menu['toppings'])}

# Extract flavor, topping, and sales from training data
X = []
sales = []

for flavor, topping, sale in menu_data:
    flavor_idx = flavor_map[flavor]
    topping_idx = topping_map[topping]
    X.append([flavor_idx, topping_idx])  # Combine flavor and topping as a single input
    sales.append(sale)  # Sales are the output

X = np.array(X)
sales = np.array(sales)

# Define the model
model = models.Sequential()
model.add(layers.InputLayer(input_shape=(2,)))  # flavor and topping each have 1 index, so input_shape=(2,)
model.add(layers.Dense(10, activation='relu'))  # Hidden layer with 10 neurons
model.add(layers.Dense(1))  # Output layer, predicting sales

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, sales, epochs=2, verbose=1)

# Predict sales
predicted_sales = model.predict(X)

# Extract flavors, toppings, actual sales, and predicted sales
flavors = [menu['flavors'][x[0]] for x in X]
toppings = [menu['toppings'][x[1]] for x in X]

# Prepare data for 3D plot
x = np.array([flavor_map[flavor] for flavor in flavors])
y = np.array([topping_map[topping] for topping in toppings])
z_actual = sales  # Actual sales
z_predicted = predicted_sales.flatten()  # Predicted sales

# Set up the 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot actual sales as a 3D scatter plot
scat_actual = ax.scatter(x, y, z_actual, c='r', marker='o', label='Actual Sales')

# Plot predicted sales as a 3D scatter plot
scat_predicted = ax.scatter(x, y, z_predicted, c='b', marker='^', label='Predicted Sales')

# Set axis labels
ax.set_xlabel('Flavors')
ax.set_ylabel('Toppings')
ax.set_zlabel('Sales')

# Set x and y axis labels
ax.set_xticks(np.arange(len(menu['flavors'])))
ax.set_xticklabels(menu['flavors'])
ax.set_yticks(np.arange(len(menu['toppings'])))
ax.set_yticklabels(menu['toppings'])

# Set graph title
ax.set_title("Actual vs Predicted Sales based on Flavor and Topping")

# Add a legend
ax.legend()

# Save the graph image
img_folder = 'img'
if not os.path.exists(img_folder):  # Create 'img' folder if it doesn't exist
    os.makedirs(img_folder)

# Save the graph image to the 'img' folder
plt.savefig(f'{img_folder}/sales_actual_vs_predicted_3d.png')

# Display the graph
plt.show()

# Save the actual vs predicted data to a DataFrame
df = pd.DataFrame({
    'Flavor': flavors,
    'Topping': toppings,
    'Actual Sales': z_actual,
    'Predicted Sales': z_predicted
})

# Create 'db' folder if it doesn't exist
db_folder = 'db'
if not os.path.exists(db_folder):
    os.makedirs(db_folder)

# SQLite 데이터베이스 연결
db_path = f'{db_folder}/sales_actual_vs_predicted.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 테이블이 없으면 생성하기
cursor.execute('''
CREATE TABLE IF NOT EXISTS sales_actual_vs_predicted (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Flavor TEXT,
    Topping TEXT,
    Actual_Sales INTEGER,
    Predicted_Sales INTEGER
)
''')

# DataFrame의 데이터를 데이터베이스에 삽입하기
for index, row in df.iterrows():
    cursor.execute('''
    INSERT INTO sales_actual_vs_predicted (Flavor, Topping, Actual_Sales, Predicted_Sales)
    VALUES (?, ?, ?, ?)
    ''', (row['Flavor'], row['Topping'], row['Actual Sales'], row['Predicted Sales']))

# 커밋 후 연결 종료
conn.commit()
conn.close()

print("데이터베이스에 성공적으로 저장되었습니다!")
