from random import seed
from random import randrange
from csv import reader
from math import sqrt

def load_csv(filename):
    #init the dataset as a list
    dataset = list()
    #open it as a readable file
    with open(filename, 'r') as file:
        #init the csv reader
		csv_reader = reader(file)
        #for every row in the dataset
        for row in csv_reader:
            if not row:
                continue
            #add that row as an element in our dataset list (2D Matrix of values)
            dataset.append(row)
    return dataset


filename = 'german.data-numeric'
dataset = load_csv(filename)