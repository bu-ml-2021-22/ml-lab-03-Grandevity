# Machine Learning Lab 3

Before you do this lab, please ensure that you have done the previous labs. Solutions are available on Brightspace.

The aims for this lab are:

* More visualisation with Matplotlib
* Visualisation with Seaborn
* Performing correlation analysis
* Visualising a correlation matrix
* Analysing data quality and cleaning data with regards to:
  - Duplicates
  - Noise / Outliers
  - Missing values

## 1 - Exploring the Iris Dataset

This dataset is provided in a ``.arff`` format. It's essentially a CSV format with meta-data (annotated). Feel free to open it up in a text editor and have a look :wink:

### 1.1 - Loading the dataset

The Jupyter Notebook for this lab already loads the dataset, using ``SciPy`` to load from the ``.arff`` format.

```python
data = arff.loadarff('data/iris.arff')
```

Then, creating the Pandas DataFrame for us to work with.

```python
iris_df = pd.DataFrame(data[0])
```

This process has loaded the class names (string) as byte strings, so the following line of code has been added to fix that encoding.

```python
iris_df['class'] = iris_df['class'].str.decode('utf-8') # fixes byte strings, avoiding strings like b'Iris-versicolor'
```

### 1.2 - Dataset information / basic stats

Using what you learnt last week, please answer the following questions:

**QUESTION 1.2.1**: How many instances are there in the dataset?

**QUESTION 1.2.2**: How many features are in the dataset (excluding the class)?

**QUESTION 1.2.3**: What type of features are they?

**QUESTION 1.2.4**: What's the basic, descriptive, statistics for the features?

**QUESTION 1.2.5**: How many classes are there?

**QUESTION 1.2.6**: What are the classes called?

**QUESTION 1.2.7**: How many instances are there of each class?


### 1.3 - Basic dataset plots

Following on from the question above.

**TASK 1.3.1**: Plot the class distribution as a ``bar`` chart (have a look at last week's lab if you've forgotten how to do that)

As a reference, you may find the following two pages useful:

* https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
* https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
* https://matplotlib.org/3.1.1/tutorials/index.html

You'll notice that there's some basic things you can do through ``Pandas`` directly (though it uses ``Matplotlib`` under the bonnet). But if you want to start to tailor and do a bit more complex visualisations, you need to combine this with some general ``Matplotlib`` code.

PS: for the next 2 tasks, you may find the overview of keyword arguments [[here](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#other-plots)] useful.

**TASK 1.3.2**: Plot the class distribution as a ``horizontal bar`` chart (have a look at the API page for [DataFrame.plot()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html), particularly the ``kind`` argument).

**TASK 1.3.3**: Plot the class distribution as a ``pie`` chart (have a look at the API page for [DataFrame.plot()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html), particularly the ``kind`` argument).

**TASK 1.3.4**: Give one of your bar charts a title and labels for the x and y axes have a look at the API page for [DataFrame.plot()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html) for the arguments you need to set).

There are actually different ways of doing this.
* The core Matplotlib way
* The pandas way

In the core [Matplot lib way](https://matplotlib.org/3.1.1/tutorials/introductory/usage.html), you'd do something like:

```python
import matplotlib.pyplot as plt

plt.figure()  # an empty figure with no axes
plt.suptitle('TITLE')  # add a title
plt.xlabel('X AXIX LABEL')
plt.ylabel('Y AXIX LABEL')

# code to plot, like you did above, e.g.,
iris_df['class'].value_counts().plot(kind="barh");
```

In the pandas plot function, you can do this:

```Python
import matplotlib.pyplot as plt

plt.figure()  # an empty figure with no axes
iris_df['class'].value_counts().plot(kind="barh", title='Class Distribution', xlabel='Frequency');
```

PS: for barh, x and y labels are the same in pandas

### 1.4 - Visualising feature data

Let's do some ``histogram`` plots and start working with sub-plots.

**TASK 1.4.1**: Create a ``histogram`` plot for the sepal length feature.

This was part of the lab last week, so just a reminder of something you can update:

```python
dataset['COLUMN_NAME'].plot(kind='hist', title='TITLE_OF_YOUR_CHART');
```

**TASK 1.4.2**: Change your figure to have 2 sub plots and add a ``histogram`` plot for the sepal width feature.

To do this, the first is the set-up of the figure and the [axes](https://matplotlib.org/3.1.1/tutorials/introductory/usage.html#axes):

```python
fig = plt.figure();

# Setting up figure with two sub-plots
fig, axs = plt.subplots(nrows=1, ncols=2)
```

Then to plot to a particular axis, we need to set the ``ax`` argument in the plot function, e.g.,

```python
iris_df['sepallength'].plot(ax=axs[0], kind='hist', title='Sepal length');
```

Now, do the same to add a line to plot the sepal width histogram, changing the index of the axis like when you work with bog standard arrays.

PS: for an example of a 2 x 2 grid, have a look at the solutions from lab 2.

PS: the sub-plots probably look a bit squashed, but we can change the figure size, e.g., as follows:

```python
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6)) # adds figure size
```

**TASK 1.4.3**: Plot an overlay ``histogram`` for all features.

Now we're getting to some of the neater aspects of Pandas and Matplotlib, where we can do some more complex visualisations with one line of code.

For example:

```python
iris_df.plot(kind='hist', alpha=0.5);
```

**TASK 1.4.4**: Plot a ``scatter matrix`` for all features.

In lab 2, you should have done a scatter plot. So now, let's look at how we can do a ``scatter matrix``, which can be really useful and save a lot of time.

```python
from pandas.plotting import scatter_matrix

scatter_matrix(iris_df, alpha=0.2, figsize=(16, 9), diagonal='kde');
```

We even get the histograms down the diagonal :)

That's quite neat, but... We are ultimately interested in applying machine learning to this dataset (soon!), so let's do something else to see how we're likely to have some challenges.

**TASK 1.4.5**: plot a simple scatter plot for sepal length vs sepal width

```python
iris_df.plot(kind='scatter', x='sepallength', y='sepalwidth', title='Sepal length vs width');
```

**TASK 1.4.6**: plot a colour-coded scatter plot for sepal length vs sepal width

The code above essentially does a scatter plot of all 150 instances in the dataset; i.e., mushing together all instances across the 3 classes.

To get some real insights, we need to colour code the instances belonging to each class. To do that, just copy paste the following into a cell in your jupyter notebook:

```python
fig = plt.figure();
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

# group the data instances according to the different classes they belong to
groups = iris_df.groupby('class')

# iterate over each group and plot (scatter)
for name, group in groups:
    ax.plot(group.sepallength, group.sepalwidth, marker='x', linestyle='', ms=12, label=name)

# set labels for the y and x axes
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')

# add legend to the figure
ax.legend()
```

**QUESTION 1.4.7**: Which two classes will be difficult/impossible to separate, based on the sepal length and sepal width.

Need quite a lot of code to make that happen..

What if there was a simpler way?!

Oh, wait, there is...

Let us welcome ``Seaborn`` to our Jupyter Notebook.

```python
import seaborn as sns
sns.lmplot(data=iris_df, x="sepallength", y="sepalwidth", hue='class')
```

**TASK 1.4.8**: using Seaborn, do a scatter plot (as above), but for petal length and petal width.

**QUESTION 1.4.9**: How easy may it be to separate the three classes according to these two features?

Ok, back to manually plotting pairs of things. Surely Seaborn can do a nice colour coded version of the ``scatter matrix`` we did above with ``Matplotlib``?!

```python
sns.pairplot(data=iris_df, hue="class")
```

You can find lots of information and examples of Seaborn visualisations [[here](https://seaborn.pydata.org/tutorial.html)][[here](https://seaborn.pydata.org/examples/)].

PS: Seaborn is great, but you do need to understand what it is you're visualising. Nice looking visualisations are **worthless** if they a) don't actually answer any questions, and b) you can't explain what the visualisation actually shows.


### 1.5 - Noise

Last thing on the Iris dataset for now. Let's look at noise and plotting error bars to visualise noise in the dataset.

**QUESTION 1.5.1**: Is there noise in the dataset? If so, which feature(s) are affected?

Visually, you can get a good idea of whether there's noise or not via a ``box plot``:

```python
plt.figure();

# setting up custom colouring
color = {'boxes': 'DarkGreen', 'whiskers': 'DarkOrange', 'medians': 'DarkBlue', 'caps': 'Gray'}

# a box plot for all features
box_plot = iris_df.plot.box(color=color, sym='r+');
box_plot.set_title('Box plot with custom colours')
box_plot.set_ylabel('Cm')
```

Or, using Seaborn:

```python
sns.boxplot(data=iris_df, orient="h", palette="Set2")
```

### 1.6 - Correlation Analysis

To calculate the correlation coefficients between two variables, we can use ``pearsonr`` and ``spearmanr`` from ``scipy.stats``.

For example:

```python
from scipy.stats import pearsonr

corr, _ = pearsonr(iris_df['sepallength'], iris_df['sepalwidth'])
print('Sepal length & sepal width: %.3f' % corr)
```

**TASK 1.6.1**: look at the [API reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) for using Spearman's to compute the correlation coefficient between the same features.

Conveniently, we can do this for all features in a Pandas DataFrame:

```python
# Compute pairwise correlation of all features

# Using Pearson's (linear relationship) - assumes normal distribution
corr = iris_df.corr(method='pearson')
print (corr)

# Using Sparman's (non-linear relationship) - doesn't assume normal distribution
corr = iris_df.corr(method='spearman')
print ("\n",corr)
```

We can also visualise a correlation matrix using Seaborn:

```python
# Correlation matrix - Seaborn heatmap
sns.heatmap(data=corr, annot=True, linewidths=.5, fmt= '.1f')
```

**QUESTION 1.6.2**: which two features have the strongest correlation?

It's not the greatest with larger datasets, so worth noting that we can customise the correlation matrix, e.g.,

```Python
# Correlation matrix - customisations
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
ax
```

Still, we can improve on the visualisation further from [Drazen Zaric](https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec), like we saw in the lecture:

```python
# Correlation matrix -
import heatmap
heatmap.corrplot(corr)
```

This uses the ``heatmap.py`` file that's in this repository.

You will probably find this the most useful when analysing your assignment dataset ;)


## 2 - Exploring the IMDB Movies Dataset

For the last part of this lab, we're going to work with the IMDB top 100 movies dataset (not up-to-date...).

First, using what you've learnt so far, load the ``imdb.csv`` dataset (like the heart disease dataset from the previous lab).

Then, look at the ``.head()``, ``.info()`` and ``.describe()`` data to get to know the dataset a bit.

## 2.1 - Duplicates

**QUESTION 2.1.1**: Does the dataset contain duplates? If so, how many rows?

To determine if there are duplicate instances (rows) in the dataset, we can use the [duplicated](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html) function.

```python
# Get a Pandas Series (vector) indicating TRUE for rows that are a duplicate
imdb_df.duplicated() # assuming you named your dataframe 'imdb_df'
```

Perhaps more useful is to use this and count the results:

```python
# Counting the True/False entries - should show 3 duplicates (True)
imdb_df.duplicated().value_counts()
```

And, then get the actual rows that are duplicates:

```python
duplicate_rows = imdb_df[imdb_df.duplicated()]
duplicate_rows
```

If you want to delete the duplicate entries, it's really quite simple:

```Python
# Removing the duplicate rows
imdb_df = imdb_df.drop_duplicates()

imdb_df.shape # should now show 1000 instances
```

## 2.2 - Missing Values

As with duplicates, we need to go through a few steps to get information about missing values.

### 2.2.1 - Missing Values Stats

The first step, that gives us the basis for further analysis is to get a matrix (DataFrame) that indicates whether information is missing or not:

```python
imdb_df.isnull()
```

To then get the number of missing entries per feature (column):
```python
# PS: imdb_df.info() actually gives us information about this too
imdb_df.isnull().sum()
```

**QUESTION 2.2.1**: which features (columns) have missing values?

**QUESTION 2.2.2**: how many missing values are there?


### 2.2.2 - Processing Missing Values

Most machine learning algorithms are unable to handle missing values. Actually, none of the implementations in ``scikit-learn`` can handle missing values.

So, we need to deal with the missing values. Here we'll look at two ways of dealing with missing values:

* Dropping all rows with missing values
* Imputing missing values (set to the mean for each respective feature/column)

For the former:

```python
# Dropping all rows with missing values
imdb_df_dropna = imdb_df.dropna()
```

For the latter:

```python
# Imputing missing values, using the mean
imdb_df_mean = imdb_df.fillna(imdb_df.median())
```

**QUESTION 2.2.3**: how many instances are we left with in the dataset if we drop all instances (rows) with missing values (``.dropna()``)?

### 2.2.3 - Further analysis [OPTIONAL]

This will be relevant to your assignment, so just putting that out there... :hand_over_mouth:

In the example above, we were able to find out how many missing values there are (as a general count) and which features that were affected.

Can you also find out how many rows / instances of the dataset has got missing values?

For example, where there are missing values for more than one feature, do we have cases where that happens for the same instance (row)?

### 3 - ML assessment analysis [OPTIONAL]

Now you've learned how to do more data analysis, e.g., the correlation analysis, you can optionally have a look at the Machine Learning assessment dataset we started looking at last week.

Some questions you can explore:

* Are there any strong correlations in the dataset?
* Are there any independent (non-correlated) features in the dataset?
* What is the strongest **positive** correlation with ``total mark`` (ignoring the other mark features and class)?
* What is the strongest **negative** correlation with ``total mark`` (ignoring the other mark features and class)?
* Does the data suggest that doing the formative assessment is valuable (likely to yield a higher ``total mark``)?


## Study Progress

Please report your engagement with this lab (even if you do this at a later point in time)
 * https://cispr.bournemouth.ac.uk/
 * CODE: ML_S_15_39249286
