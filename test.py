import csv

def add_index_to_csv(input_file, output_file):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    j = -1
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for i, row in enumerate(data):
            if len(row) != 0:
                writer.writerow(["index_"+str(j)] + row)
                j+=1
            else:
                writer.writerow([])

# Example usage:
input_file = 'answer1.csv'
output_file = 'output.csv'
add_index_to_csv(input_file, output_file)
