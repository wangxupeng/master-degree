import numpy as np
import matplotlib.pylab as plt
import warnings
import pandas as pd

if __name__ == '__main__':
    warnings.simplefilter('ignore')
    df = pd.read_csv(r'D:\references\Python_Data_Visualization_Cookbook_2nd-master\Chapter_03\ch03-energy-production.csv')

    columns = ['Coal', 'Natural Gas (Dry)', 'Crude Oil', 'Nuclear Electric Power',
               'Biomass Energy', 'Hydroelectric Power', 'Natural Gas Plant Liquids',
               'Wind Energy', 'Geothermal Energy', 'Solar/PV Energy']


    colors = ['darkslategray', 'powderblue', 'darkmagenta', 'lightgreen', 'sienna',
              'royalblue', 'mistyrose', 'lavender', 'tomato', 'gold']

    plt.figure(figsize=(12,8))
    polys = plt.stackplot(df['Year'], df[columns].values.T, colors=colors)

    rectangles = []
    for poly in polys:
        rectangles.append(plt.Rectangle((0,0), 1, 1, fc=poly.get_facecolor()[0]))
    legend = plt.legend(rectangles, columns, loc=3)
    frame = legend.get_frame()
    frame.set_color('white')

    plt.title('Primary Energy Production by Source', fontsize = 16)
    plt.xlabel('Year', fontsize = 16)
    plt.ylabel('Production (Quad BTU)', fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xlim(1973, 2014)

    plt.show()

