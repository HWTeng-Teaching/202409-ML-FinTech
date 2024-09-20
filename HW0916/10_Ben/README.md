import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
college = pd.read_csv("College.csv")


college2 = pd.read_csv('College.csv', index_col=0)
college3 = college.rename({'Unnamed: 0': 'College'},
axis=1)
college3 = college3.set_index('College')

college = college3

summary = college.describe()

# Select the desired columns for the scatterplot matrix
columns = ['Top10perc', 'Apps', 'Enroll']
selected_data = college[columns]

# Create the scatterplot matrix
scatter_matrix = pd.plotting.scatter_matrix(selected_data, figsize=(10, 10))

# Add labels to the diagonal subplots
for ax in scatter_matrix.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize = 12)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 12)

plt.suptitle('Scatterplot Matrix of Top10perc, Apps, Enroll', y=0.95, fontsize=16)
plt.show()

# Create side-by-side boxplots of 'Outstate' versus 'Private'
college.boxplot(column='Outstate', by='Private', figsize=(8, 6))

# Set the title and labels
plt.title('Boxplot of Outstate vs Private')
plt.xlabel('Private')
plt.ylabel('Outstate')

# Show the boxplot
plt.show()

# Create a new column 'Elite' based on the Top10perc variable
college['Elite'] = pd.cut(college['Top10perc'], [0, 50, 100], labels=['No', 'Yes'])

# Count the number of elite universities
elite_counts = college['Elite'].value_counts()
print(elite_counts)

# Create side-by-side boxplots of 'Outstate' versus 'Elite'
college.boxplot(column='Outstate', by='Elite', figsize=(8, 6))

# Set the title and labels
plt.title('Boxplot of Outstate vs Elite')
plt.xlabel('Elite')
plt.ylabel('Outstate')

# Show the boxplot
plt.show()

# Select a few quantitative variables for which you want to create histograms
selected_vars = ['Apps', 'Accept', 'Enroll', 'Top10perc']

# Create subplots for multiple histograms
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Generate histograms with different numbers of bins
for i, var in enumerate(selected_vars):
    row = i // 2
    col = i % 2
    college[var].plot.hist(ax=axs[row, col], bins=10*(i+1), edgecolor='black')
    axs[row, col].set_title(f'Histogram of {var} with {10*(i+1)} bins')
    axs[row, col].set_xlabel(var)
    axs[row, col].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
