import os
import pickle

dict_of_models = {}
model_folder = os.path.join(os.path.dirname(os.getcwd()), r"preTrainedModels")
for model_name in os.listdir(model_folder):
     if model_name.endswith(".pkl"):
        with open(os.path.join(model_folder, model_name), "rb") as f:
            data = pickle.load(f)
            dict_of_models[os.path.splitext(model_name)[0]] = data
            
print(dict_of_models)
