import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('digital_resources.csv')

# Preprocessing
data_cleaned = preprocess(data)

# Clustering
kmeans = KMeans(n_clusters=3).fit(data_cleaned[['Usability', 'Scalability']])
data['Cluster'] = kmeans.labels_

# Regression
model = LinearRegression()
model.fit(data[['Cost', 'Reliability']], data['Performance'])

# Visualization
plt.scatter(data['Usability'], data['Performance'], c=data['Cluster'])
plt.xlabel('Usability')
plt.ylabel('Performance')
plt.title('Clustering of Digital Resources')
plt.show()
