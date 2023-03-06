import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pickle

def load_all(model_folder):
	global dict_of_models, list_of_models
	dict_of_models = {}
	list_of_models = []
	for model_name in os.listdir(model_folder):
		if model_name.endswith(".pkl"):
			with open(os.path.join(model_folder, model_name), "rb") as f:
				data = pickle.load(f) # lol
				dict_of_models[os.path.splitext(model_name)[0]] = data
	
	for model_name in dict_of_models:
		list_of_models.append(model_name)
	
	return list_of_models, dict_of_models

