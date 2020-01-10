import pandas
import random

NUMBER_OF_CITIES = 100
MIN_COST = 1
MAX_COST = 100


def generate_cell_value(row, col):
    if row == 0 and col == 0:
        return -1
    elif row == 0:
        return col - 1
    elif col == 0:
        return row - 1
    elif row == col:
        return 0
    else:
        return random.randint(MIN_COST, MAX_COST)


data_table = [[generate_cell_value(i, j) for i in range(NUMBER_OF_CITIES+1)] for j in range(NUMBER_OF_CITIES+1)]

df = pandas.DataFrame(data_table)
writer = pandas.ExcelWriter('salesman_problem.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', header=None, index=False)
writer.save()

