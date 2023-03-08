import pathlib
import os


def find_models_folder(path_to_find):
    pure_path = pathlib.Path(__file__)
    un_pure_path = pure_path.as_posix()
    for i in pure_path.parents:
        if i.is_dir():
            print(i)
            files = i.glob('*')
            dir_list = [file for file in files if file.is_dir()]
            for dirs in dir_list:
                if dirs.name == path_to_find:
                    return os.path.join(i, dirs.name)
    return None


#print(find_models_folder("preTrainedModels"))
    


exit()
from models import load_all

print(os.path.join(os.path.dirname(os.getcwd()), r"preTrainedModels"))
list_of_models, dict_of_models = load_all(os.path.join(os.path.dirname(os.getcwd()), r"preTrainedModels"))
print(list_of_models)
print(dict_of_models)