import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    plt.figure(figsize=(12,8))
    y = np.random.randn(10)

    col_labels = ['col1', 'col2', 'col3']
    row_labels = ['row1', 'row2', 'row3']
    table_vals = [[11, 12, 13], [21, 22, 23], [28, 29, 30]]
    row_colors = ['red', 'gold', 'green']
    my_table = plt.table(cellText = table_vals,
                         colWidths= [0.1] * 3,
                         rowLabels = row_labels,
                         colLabels = col_labels,
                         rowColours = row_colors,
                         loc = 'upper right')

    plt.plot(y)
    plt.show()
