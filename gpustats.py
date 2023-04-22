#import sys
import subprocess
from time import sleep
import os
clear = lambda: os.system('clear')

for i in range(500):
    sleep(100)
    #clear()
    subprocess.run('gpustat')
    #os.system('cls')
    #print(str(type(out)))