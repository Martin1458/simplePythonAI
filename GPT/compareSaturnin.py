import os
import re

direct = os.getcwd()
pth_names = ["GPT_saturninV2", "GPT_saturninV2New"]

def get_pth():
    someRandomAssList = []
    for fn in os.listdir(direct):
        if any(fn.startswith(name) for name in pth_names) and fn.endswith(".pth"):
            someRandomAssList.append(fn)
    
    return someRandomAssList

def sepatare_pth(p):
    allListGod = []
    for item in pth_names:
        allListGod.append([])
    """
    smList = []
    for prefix in pth_names:
        sublist = []
        for pth_name in p:
            if pth_name.startswith(prefix) and all(pth_name != other_name for other_name in p if pth_name.startswith(prefix) and other_name != prefix):
                sublist.append(pth_name)
        smList.append(sublist)
    
    return smList
    """
    """
    separated = [[pth_name for pth_name in p if pth_name.startswith(prefix) and all(pth_name)] for prefix in pth_names]
    """
    
    return separated

def get_txt():
    listek = []
    for filename in os.listdir(direct):
        if filename.startswith("GPT_saturnin") and filename.endswith(".txt"):
            listek.append(filename)
    
    return listek

def get_steps():
    # StepTrainlossValloss
    STV = {}
    txt_list = get_txt()

    for txt_file in txt_list:
        with open(txt_file, "r") as f:
            all_lines = []
            for line in f:
                if line.startswith("step"):
                    nums_in_line = re.findall(r"\d+(?:\.\d+)?", line)
                    nums_as_nums = [int(nums_in_line[0]) + 1 if nums_in_line[0][-1] == "9" else int(nums_in_line[0]) , float(nums_in_line[1]), float(nums_in_line[2])]
                    all_lines.append(nums_as_nums)
            STV[str(txt_file)] = all_lines
    
    return STV


the_dict = get_steps()
pth_files = get_pth()
pth_sorted = sepatare_pth(pth_files)
print("GPT/compareSaturnin.py:52 pth_sorted:", pth_sorted)

