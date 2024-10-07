import csv

filename = '/home/belief/Desktop/MLProj1/mini-project-1/datasets/train/train_text_seq.csv'
column_index = 1
value_1_count = 0
value_0_count = 0

with open(filename, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    
    for row in reader:
        if row[column_index] == '1':
            value_1_count += 1
        elif row[column_index] == '0':
            value_0_count += 1

print(f"Number of instances with value 1: {value_1_count}")
print(f"Number of instances with value 0: {value_0_count}")