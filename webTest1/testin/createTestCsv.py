import random
import csv

myList = []

for i in range(10):
    x = random.randint(0, 100)
    y = random.randint(0, 100)
    if x == y:
        myList.append([x, y, x])
    elif x > y:
        myList.append([x, y, x])
    elif y > x:
        myList.append([x, y, y])
    else:
        print("fuck this")

print(myList)

with open("testin.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(myList)
