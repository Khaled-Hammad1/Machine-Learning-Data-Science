from KNN import KNN
from RandomForest import RandomForest
from CNN import CNN

while True:
    x = input("1=CNN, 2=RandomForest, 3=KNN, 0=Exit: ").strip()

    if x == "1":
        CNN()
    elif x == "2":
        RandomForest()
    elif x == "3":
        KNN()
    elif x == "0":
        break
    else:
        print("Invalid choice, try again.")
