import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

input_list = np.random.randint(0, 4, (5000, 2))

sum_input_list = np.sum(input_list, axis=1)

output_list = np.array([[sum_input_list[i]] for i in range(sum_input_list.shape[0])])

input_list_train, input_list_test, output_list_train, output_list_test = train_test_split(input_list, output_list, test_size=0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(input_list, output_list, epochs=10, batch_size=4)

new_input = np.array([[80, 70], [50, 30], [90, 100]])
predictions = model.predict(new_input)
print("{:<10s}{:<15s}{}".format(" ", "prediction", "actual"))
for prediction in range(len(predictions)):
    print("{:<10s}{:<15s}{}".format(str(new_input[prediction][0])+"+"+str(new_input[prediction][1])+":", str(predictions[prediction][0]), np.sum(new_input[prediction])))

model.evaluate(input_list_test, output_list_test)

import pickle
import os

pkFile = r"NN-Sum.pkl"
if not os.path.exists(pkFile): 
    with open(pkFile, 'wb') as f:
        pickle.dump(model, f)

print(pkFile)
with open(pkFile, "rb") as f:
    data = pickle.load(f)
    print("data type"+str(type(data)))
    numbers = []
    prediction = data.predict([1, 1])
    print(prediction)

