from testData import input_list, output_list
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Split the data into training and testing sets
input_list_train, input_list_test, output_list_train, output_list_test = train_test_split(input_list, output_list, test_size=0.2)

# Train the algorithm
reg = LinearRegression().fit(input_list_train, output_list_train)

# Evaluate the algorithm
accuracy = reg.score(input_list_test, output_list_test)
print("Accuracy: ", accuracy)

# Use the trained algorithm to make predictions
input_string = input("Enter two numbers separated by a space: ")
input_numbers = input_string.split()
prediction = reg.predict(np.array([list(map(int, input_numbers))]))
print("Prediction: ", prediction)