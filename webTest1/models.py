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

#load_model("NN-Sum.pkl")
print(os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "NN-Sum.pkl")))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "NN-Sum.pkl"), "rb") as f:
	data = pickle.load(f)

dict_of_models = {}
list_of_models = []

def load_all():
	global dict_of_models, list_of_models
	dict_of_models = {}
	list_of_models = []
	model_folder = os.path.join(os.path.dirname(os.getcwd()), r"preTrainedModels")
	for model_name in os.listdir(model_folder):
		if model_name.endswith(".pkl"):
			with open(os.path.join(model_folder, model_name), "rb") as f:
				data = pickle.load(f)
				dict_of_models[os.path.splitext(model_name)[0]] = data
	
	for model_name in dict_of_models:
		list_of_models.append(model_name)
	
	return list_of_models, dict_of_models

