# Run using: uvicorn main:app --reload 
# Make sure you are in the same directory as this file or youll get the Error loading ASGI app.
from fastapi import FastAPI, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
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

list_of_models = None

@app.get("/get_models")
async def get_models():
    return list_of_models

@app.get("/get_prediction")
async def get_prediction(myModel: str, myData: str):
    if myModel == "" or myData == "":
        return "No"
    try:
        myNewData = myData.split(" ")
        myNewData = [[ int(i) for i in myNewData]]
        
        print("myModel, myNewData"+ myModel, myNewData)
        selectedModel = dict_of_models[myModel]
        prediction = selectedModel.predict(myNewData)
        
        return str(prediction)
    except Exception as e:
        return e


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}

from webTest1.models import load_all, find_models_folder
print("1")
# find models folder
models_folder = find_models_folder("preTrainedModels")
print("1")
error = True if models_folder == None else False
print("2")
if not error:
    # setup models
    print("2noError")
    #list_of_models, dict_of_models = load_all(os.path.join(os.path.dirname(os.getcwd()), r"preTrainedModels"))
    list_of_models, dict_of_models = load_all(models_folder)
    print(list_of_models)
    print(dict_of_models)
    selectedModel = dict_of_models['NN-Sum']
    prediction = selectedModel.predict([[1,5]])
    print(prediction)

print("done")