import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
import seaborn as sns
#loading dataset
# load the iris.arff data set
data = arff.loadarff('data/iris.arff')
iris_df = pd.DataFrame(data[0])
iris_df['class'] = iris_df['class'].str.decode('utf-8') # fixes byte strings, avoiding strings like b'Iris-versicolor'


#1.5 Noise
#Box plot in matplotlib/pandas
plt.figure();

# setting up custom colouring
color = {'boxes': 'DarkGreen', 'whiskers': 'DarkOrange', 'medians': 'DarkBlue', 'caps': 'Gray'}

# a box plot for all features
box_plot = iris_df.plot.box(color=color, sym='r+');
box_plot.set_title('Box plot with custom colours')
box_plot.set_ylabel('Cm')
#and... using Seborn
sns.boxplot(data=iris_df, orient="h", palette="Set2")
# Calculate correlation coefficient sepal length and sepal width
from scipy.stats import pearsonr

corr, _ = pearsonr(iris_df['sepallength'], iris_df['sepalwidth'])
print('Sepal length & sepal width: %.3f' % corr)
plt.show()

# Task 1.6.1 - use Spearman to calculate the correlation coefficient too

# Compute pairwise correlation of all features

# Question 1.6.2 - which features have the strongest correlation?

# Correlation matrix - Seaborn heatmap

# Correlation matrix - customisations

# Correlation matrix - https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec

# imdb_df = ...



#Duplicates

# Get a Pandas Series (vector) indicating TRUE for rows that are a duplicate

# Counting the True/False entries - should show 3 duplicates (True)

# Get the actual rows that are duplicates

# Removing the duplicate rows

# Get DataFrame indicating True/False for whether values are missing or not

# Get the number of missing entries per feature (column)

# Dropping all rows with missing values

# Imputing missing values, using the mean

