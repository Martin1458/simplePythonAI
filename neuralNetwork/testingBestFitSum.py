# %% [markdown]
# Import important modules

# %%
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# %% [markdown]
# Creating data

# %%
input_list = np.random.randint(0, 40000, (5000, 2))

sum_input_list = np.sum(input_list, axis=1)

output_list = np.array([[sum_input_list[i]] for i in range(sum_input_list.shape[0])])

# %% [markdown]
# Splitting data into training and testing

# %%
input_list_train, input_list_test, output_list_train, output_list_test = train_test_split(input_list, output_list, test_size=0.2)

# %% [markdown]
# Defining keras model

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# %% [markdown]
# Compiling the model

# %%
model.compile(optimizer='adam', loss='mean_squared_error')

# %% [markdown]
# Training the model

# %%
myPlotData = []
for epoch in range(2, 22, 2):
    for batch_size in range(2, 22, 2):
        model.fit(input_list, output_list, epochs=epoch, batch_size=batch_size)
            
        loss = model.evaluate(input_list_test, output_list_test)
        score = 1 / (1 + loss)  # Convert the loss to a score
        
        myPlotData.append([epoch, batch_size, score])
print(myPlotData)


# %% 
sorted_data = sorted(myPlotData, key=lambda x: x[2], reverse=True)
for dat in sorted_data:
    print(dat)

# %%
import numpy as np
import matplotlib.pyplot as plt
 
 
fig = plt.figure()
ax = plt.axes(projection ='3d')

epochs = []
batch_sizes = []
scores = []

for data in sorted_data:
    epochs.append(data[0])
    batch_sizes.append(data[1])
    scores.append(data[2])
    
ax.plot3D(epochs, batch_sizes, scores, 'green')
ax.set_title('3D')
plt.show()

# %% [markdown]
# Check if this model is already exported, if not export it

# %%
import pickle
import os
import pathlib

oneUp = pathlib.Path(os.path.dirname(os.getcwd()))
preModels = pathlib.Path('preTrainedModels')
pkFile = oneUp.joinpath(preModels).joinpath("NN-Sum.pkl")
print(pkFile)
#pkFile = os.path.join(os.path.dirname(os.getcwd()), r"preTrainedModels\NN-Sum.pkl")
print(os.path.exists(pkFile))
with open(pkFile, 'wb') as f:
    pickle.dump(model, f)

# %%
with open(pkFile, "rb") as f:
    data = pickle.load(f)
    print("data type"+str(type(data)))
    numbers = np.array([[15, 15]])
    prediction = data.predict(numbers)
    print(prediction)


