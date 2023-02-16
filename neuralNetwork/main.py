import numpy as np
import tensorflow as tf
from testData import input_list, output_list
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
input_list_train, input_list_test, output_list_train, output_list_test = train_test_split(input_list, output_list, test_size=0.2)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(input_list, output_list, epochs=10, batch_size=4)

# Make predictions on new input
new_input = np.array([[80, 70], [50, 30], [90, 100]])
predictions = model.predict(new_input)
print(predictions)
exit()
# Print the predictions
for prediction in range(predictions):
    print(prediction)
    #print(predictions[prediction], " ", np.sum(new_input[prediction]))