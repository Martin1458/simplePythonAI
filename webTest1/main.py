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
 
@app.get("/")
#async def root(numbers: list[int] = Query([])):
#    total = sum(numbers)
#    return {"message": f"The sum of {numbers} is {total}"}
async def root(myVar: str):
    try:
        myNewVar = myVar.split(" ")
        myNewVar = [ int(i) for i in myNewVar]
        myNewVar = sum(myNewVar)
        myVar = myNewVar
    except:
        pass
    return myVar