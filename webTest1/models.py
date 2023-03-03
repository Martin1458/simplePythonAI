import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

#print(script_dir)
#print(parent_dir)

models_folder = os.path.join(parent_dir, "preTrainedModels")

# Assume `X_new` is a new dataset you want to make predictions on
#y_pred = model.predict(X_new)

def load_model(model_file_name):
    print(os.path.join(models_folder, model_file_name))
    with open(os.path.join(models_folder, model_file_name), "rb") as f:
        model = pickle.load(f)
    print("Model type"+str(type(model)))
    numbers = []
    prediction = model.predict([1, 1])
    print(prediction)

load_model("NN-Sum.pkl")

