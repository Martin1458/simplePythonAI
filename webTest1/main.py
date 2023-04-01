# Run using: uvicorn main:app --reload 
# Make sure you are in the same directory as this file or youll get the Error loading ASGI app.
from fastapi import FastAPI, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from webTest1.models import load_all_models, find_folder
import os
from webTest1.creatorNN import createNN, get_tf_predictions
import pickle
import pathlib
from webTest1.funFile import get_list_of_files

app = FastAPI()
error = False

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
    
 
@app.get("/")
#async def root(numbers: list[int] = Query([])):
#    total = sum(numbers)
#    return {"message": f"The sum of {numbers} is {total}"}
async def root(myVar: str):
    if myVar == "":
        return ""
    try:
        myNewVar = myVar.split(" ")
        myNewVar = [ int(i) for i in myNewVar]
        myNewVar = sum(myNewVar)
        myVar = myNewVar
    except:
        pass
    return myVar

list_of_models = []
dict_of_models = {}
user_list_of_models = []
user_dict_of_models = {}

@app.get("/get_models")
async def get_models():
    global list_of_models, dict_of_models
    list_of_models, dict_of_models = find_models(find_folder("preTrainedModels"))
    return list_of_models


@app.get("/get_user_models")
async def get_user_models():
    global user_list_of_models, user_dict_of_models
    user_list_of_models, user_dict_of_models = find_models(find_folder("userCreatedModels"))
    return user_list_of_models
    
@app.get("/get_user_files")
async def get_user_files():
    return get_list_of_files(find_folder("userUploads"))

@app.get("/get_prediction")
async def get_prediction(myModel: str, myData: str):
    if myModel == "" or myData == "":
        print(myModel)
        print(myData)
        return "No"
    try:
        myNewData = myData.split(" ")
        myNewData = [[ int(i) for i in myNewData]]
        
        print("myModel, myNewData"+ myModel, myNewData)
        dict_of_all_models = dict_of_models | user_dict_of_models
        selectedModel = dict_of_all_models[myModel]
        prediction = selectedModel.predict(myNewData)
        
        return str(prediction)
    except Exception as e:
        return e

uploadedFiles = []
@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    global uploadedFiles
    if not os.path.isfile(file):
        try:
            uploads_folder = find_folder("userUploads")
            print("file: "+str(uploads_folder.joinpath(file.filename)))
            contents = file.file.read()
            with open(uploads_folder.joinpath(file.filename), 'wb') as f:
                f.write(contents)
            uploadedFiles.append(uploads_folder.joinpath(file.filename))
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()

        print(uploadedFiles)
        return {"message": f"Successfully uploaded {file.filename}"}
    else:
        return {"message": f"File already exists {file.filename}"}

@app.post("/createNN")
async def create_nn(name_model: str, num_epochs: str, size_of_batch: str):
    new_model = createNN(uploadedFiles[-1], num_epochs, size_of_batch)

    oneUp = pathlib.Path(os.path.dirname(os.getcwd()))
    usrModels = pathlib.Path('userCreatedModels')
    pkFile = oneUp.joinpath(usrModels ).joinpath(name_model+".pkl")
    with open(pkFile, 'wb') as f:
        pickle.dump(new_model, f)
        return "new model has been created successfully"
    pass

def find_models(models_folder):
    error = True if models_folder == None else False
    if not error:
        temp_list_of_models, temp_dict_of_models = load_all_models(models_folder)
        print("temp_list_of_models: " + str(temp_list_of_models))
        print("temp_dict_of_models: " + str(temp_dict_of_models))
    return temp_list_of_models, temp_dict_of_models

def find_user_models():
    global user_list_of_models, user_dict_of_models
    # find models folder
    user_models_folder = find_folder("userCreatedModels")
    error = True if user_models_folder == None else False
    if not error:
        # setup models
        #list_of_models, dict_of_models = load_all_models(os.path.join(os.path.dirname(os.getcwd()), r"preTrainedModels"))
        user_list_of_models, user_dict_of_models = load_all_models(user_models_folder)
        print("user_list_of_models: " + str(user_list_of_models))
        print("user_dict_of_models: " + str(user_dict_of_models))

    print("Loaded")