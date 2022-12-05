import csv

def openfile(path):
    rows = []
    with open(path, 'r') as file:
        csvreader = csv.reader(file)
        headers = next(csvreader)
        for row in csvreader:
            rows.append(row)
    return rows, len(headers)

#openfile('/data/test.csv')