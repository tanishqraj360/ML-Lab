# Find-S algo
import csv

with open("enjoysport.csv") as f:
    data = [row for row in csv.reader(f) if row and row[-1].strip().lower() == "yes"]

hypothesis = data[0][:-1]  # Initialize with first positive example
for row in data[1:]:
    for i in range(len(hypothesis)):
        if hypothesis[i] != row[i]:
            hypothesis[i] = "?"

print("Most specific hypothesis:", hypothesis)
