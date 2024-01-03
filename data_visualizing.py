import pandas as pd
### Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def create_heatmap(data, value_min, value_max
                   , y_titles, title):
    plt.figure(figsize=(12,9))
    sns.heatmap(data,
            vmin = value_min, 
            vmax = value_max,
            cmap = 'RdBu',
            annot = True)
    plt.yticks([0, 1, 2], 
           y_titles,
           rotation = 45,
           fontsize = 12)
    plt.title(title,fontsize = 14)
    plt.show()


def create_scatter(data_segm, x_col, y_col, label_col, title):
    mall_analysis = data_segm.groupby([label_col]).mean()
    x_axis = data_segm[x_col]
    y_axis = data_segm[y_col]
    plt.figure(figsize = (10, 8))
    sns.scatterplot(data=mall_analysis, x=x_axis, y=y_axis, 
                    hue=data_segm[label_col], 
                    palette=['green', 'red', 'orange', 'gray', 'purple', 'yellow'])
    plt.title(title)
    plt.show()