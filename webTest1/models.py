import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pickle
import pathlib

def find_folder(path_to_find):
    pure_path = pathlib.Path(__file__)
    un_pure_path = pure_path.as_posix()
    for i in pure_path.parents:
        if i.is_dir():
            #print(i)
            files = i.glob('*')
            dir_list = [file for file in files if file.is_dir()]
            for dirs in dir_list:
                if dirs.name == path_to_find:
                    return i.joinpath(dirs.name)
    return None

def load_all(model_folder):
	global dict_of_models, list_of_models
	dict_of_models = {}
	list_of_models = []
	for model_name in os.listdir(model_folder):
		if model_name.endswith(".pkl"):
			print(model_name)
			with open(os.path.join(model_folder, model_name), "rb") as f:
				data = pickle.load(f) # lol
				dict_of_models[os.path.splitext(model_name)[0]] = data
	
	for model_name in dict_of_models:
		list_of_models.append(model_name)
	
	return list_of_models, dict_of_models

