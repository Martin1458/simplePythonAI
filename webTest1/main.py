# Run using: uvicorn main:app --reload 
# Make sure you are in the same directory as this file or youll get the Error loading ASGI app.
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# setup models
import os
from webTest1.models import load_all
#list_of_models, dict_of_models = load_all(os.path.join(os.path.dirname(os.getcwd()), r"preTrainedModels"))
list_of_models, dict_of_models = load_all(r"C:\Users\marti\Desktop\PythonProjects\simplePythonAI\preTrainedModels")
    
 
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

@app.get("/get_models")
async def get_models():
    return list_of_models