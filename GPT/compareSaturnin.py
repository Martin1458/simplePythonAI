import os
import re
from pathlib import Path

scriptPath = Path(__file__).resolve()
direct = scriptPath.parent
# Just for testing on mu laptop ==> Delete if u dont know me
direct = Path("/home/martin/Desktop/pythonShit/webapp/simplePythonAINewnew/simplePythonAI")
directCurr = scriptPath.parent

# Define the prefixes for the .pth files
pth_names = ["GPT_saturninV2", "GPT_saturninV2New"]

# Function to get all .pth files with the specified prefixes
def get_pth():
    someRandomAssList = []
    for fn in os.listdir(direct):
        if any(fn.startswith(name) for name in pth_names) and fn.endswith(".pth"):
            someRandomAssList.append(fn)
    
    return someRandomAssList

# Function to separate .pth files based on their prefixes
def sepatare_pth(p):
    allListGod = []
    for item in pth_names:
        allListGod.append([])
    
    smList = []
    for prefix in pth_names:
        sublist = []
        for pth_name in p:
            if pth_name.startswith(prefix) and all(pth_name != other_name for other_name in p if pth_name.startswith(prefix) and other_name != prefix):
                sublist.append(pth_name)
        smList.append(sublist)
    
    return smList
    
    """
    separated = [[pth_name for pth_name in p if pth_name.startswith(prefix) and all(pth_name)] for prefix in pth_names]
    """
    
    return separated

# Function to get all .txt files with the specified prefix
def get_txt():
    listek = []
    for filename in os.listdir(directCurr):
        if filename.startswith("GPT_saturnin") and filename.endswith(".txt"):
            listek.append(filename)
    
    return listek

# Function to extract step, train loss, and val loss from .txt files
def get_steps():
    # StepTrainlossValloss
    STV = {}
    txt_list = get_txt()

    for txt_file in txt_list:
        with open(directCurr.joinpath(txt_file), "r") as f:
            all_lines = []
            for line in f:
                if line.startswith("step"):
                    nums_in_line = re.findall(r"\d+(?:\.\d+)?", line)
                    nums_as_nums = [int(nums_in_line[0]) + 1 if nums_in_line[0][-1] == "9" else int(nums_in_line[0]) , float(nums_in_line[1]), float(nums_in_line[2])]
                    all_lines.append(nums_as_nums)
            STV[str(txt_file)] = all_lines
    
    return STV


# Funtion to connect .pth files to thair steps
def connect_pth_score(scores, pth_files_list):
    txt_prefix = []
    for txt_file in scores:
        txt_prefix.append(txt_file.replace(".txt", ""))
    print(txt_prefix)
    for file in pth_files_list:
        pass

# Get step, train loss, and val loss from .txt files
the_dict = get_steps()
#print("➡ the_dict :", the_dict)
# Get .pth files with specified prefixes
pth_files = get_pth()
print("➡ pth_files :", pth_files)
connect_pth_score(the_dict, pth_files)
# Separate .pth files based on their prefixes
pth_sorted = sepatare_pth(pth_files)
#print("GPT/compareSaturnin.py:52 pth_sorted:", pth_sorted)

