#import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import csv

def createNN(fileName, num_epochs, size_of_batch):
    with open(fileName) as f:
        wholeCSV = list(csv.reader(f, delimiter=','))
        #print(wholeCSV)
    if len(wholeCSV[0]) != 3:
        return {"error":"The csv file isnt correcrly formatted"}
    input_list = []
    output_list = []
    print("type of num: "+str(type(wholeCSV[0][0])))
    intWholeCSV = [list(map(int,i)) for i in wholeCSV]
    print("type of num: "+str(type(intWholeCSV[0][0])))
    for row in intWholeCSV:
        input_list.append([row[0], row[1]])
        output_list.append([row[2]])
    print("input_list: "+ str(input_list))
    print("output_list: "+ str(output_list))

    #Split data into training and testing set
    input_list_train, input_list_test, output_list_train, output_list_test = train_test_split(input_list, output_list, test_size=0.2)
    print("input_list_train: "+ str(input_list_train))
    print("input_list_test: "+ str(input_list_test))
    print("output_list_train: "+ str(output_list_train))
    print("output_list_test: "+ str(output_list_test))

    #Create the keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    #Compile model
    model.compile(optimizer="adam", loss="mean_squared_error")

    #Trining the model
    model.fit(input_list, output_list, epochs=num_epochs, batch_size=size_of_batch)
    
    
    return {"done":model}

#model = createNN("webTest1/testin/testin.csv", 6, 4)


def get_tf_predictions(model_object, what_to_predict):
    prediction = model_object.predict(what_to_predict)
    return prediction

#if "done" in model:
    #print(get_tf_predictions(model["done"], [[1, 4]]))
