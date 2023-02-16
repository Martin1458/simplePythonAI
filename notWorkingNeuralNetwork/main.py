import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Removes the warning and info messages
#from testData import input_list, output_list
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

# Generate an array of 5 lists, each containing 2 random integers between 0 and 100 (inclusive)
input_list = np.random.randint(0, 101, (50, 2))
#print(input_list)

# Create a new array containing the sum of each list in input_list
sum_input_list = np.sum(input_list, axis=1)

# Create a list of lists, where each sublist contains a single element from sum_input_list
output_list = np.array([[sum_input_list[i]] for i in range(sum_input_list.shape[0])])
#print(output_list)


# Split the data into training and testing sets
input_list_train, input_list_test, output_list_train, output_list_test = train_test_split(input_list, output_list, test_size=0.2)
#print(input_list_train, input_list_test, output_list_train, output_list_test)

# Create a neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_list_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(input_list_train, output_list_train, epochs=100, verbose=False)

# Evaluate the model
loss = model.evaluate(input_list_test, output_list_test, verbose=False)
print("Test Loss: ", loss)

# Use the trained algorithm to make predictions
input_string = input("Enter two numbers separated by a space: ")
input_numbers = input_string.split()
prediction = model.predict(input_numbers)
print("Prediction: ", prediction)