# -*- coding: utf-8 -*-

# read excel file.
import os

import xlrd
# write to excel file.
import xlwt
# using lagrange interpolate.
from scipy.interpolate import lagrange


# define interpolate function.
def ployinterp_column(s, n, k):
    array = list(range(n - k, n)) + list(range(n + 1, n + 1 + k))
    y = []
    for i in array:
        y.append(s[i])
    return lagrange(array, y)(n)


# read excel file content.
old_data_path = '../data/catering_sale_missing.xls'
old_data = xlrd.open_workbook(old_data_path)
old_sheet = old_data.sheets()[0]
old_rows = old_sheet.nrows

# column vector.
s = []
# missing data index array.
interpolateIndexes = []

# calculate lagrange interpolate.
first_column_name = ''
second_column_name = ''
first_column_values = []
for i in range(old_rows):
    if i == 0:
        first_column_name = old_sheet.row_values(i)[0]
        second_column_name = old_sheet.row_values(i)[1]
        print(first_column_name, '      ', second_column_name)
        continue

    first_column_values.append(old_sheet.row_values(i)[0])
    v = old_sheet.row_values(i)[1]
    if v == '':
        interpolateIndexes.append(i - 1)
    s.append(v)
    print(first_column_values[i - 1], '   ', s[i - 1])

k = 3
for i in interpolateIndexes:
    s[i] = round(ployinterp_column(s, i, k), 1)

print('   ')
print('   ')
print('   ')

# write the new data to missing_data_processing.xls file.
new_data_path = '../data/missing_data_processing.xls'

if os.path.exists(new_data_path):
    os.remove(new_data_path)

new_data = xlwt.Workbook(encoding='utf-8')
new_sheet = new_data.add_sheet('Sheet1')

# define date format
date_style = xlwt.XFStyle()
date_style.num_format_str = 'YYY/M/D'

for i in range(old_rows):
    if i == 0:
        new_sheet.write(i, 0, label=first_column_name)
        new_sheet.write(i, 1, label=second_column_name)
        print(first_column_name, '      ', second_column_name)
        continue

    new_sheet.write(i, 0, first_column_values[i - 1], date_style)
    new_sheet.write(i, 1, s[i - 1])
    print(first_column_values[i - 1], '   ', s[i - 1])

new_data.save(new_data_path)

