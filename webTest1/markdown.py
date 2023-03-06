import os
from models import load_all

print(os.path.join(os.path.dirname(os.getcwd()), r"preTrainedModels"))
list_of_models, dict_of_models = load_all(os.path.join(os.path.dirname(os.getcwd()), r"preTrainedModels"))
print(list_of_models)
print(dict_of_models)