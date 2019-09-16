# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:19:22 2019

@author: Owner
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
import seaborn as sns
from sklearn.cluster import KMeans
mallcustomers = pd.read_csv('Mall_Customers.csv')

#General Information
mallcustomers.head
mallcustomers.shape
mallcustomers.isnull().any()
female= mallcustomers.loc[mallcustomers.Gender== 'Female']
male= mallcustomers.loc[mallcustomers.Gender== 'Male']

#Check gender distribution
piedata = mallcustomers.groupby(['Gender']).sum()
f, axes = mtp.subplots(1,1, figsize=(6,6))
axes.set_title("Male-Female Customer distribution")
piedata.plot(kind='pie',y=2, ax=axes, fontsize=14,shadow=False,autopct='%1.1f%%');
axes.set_ylabel('');
mtp.show()
''' More Females visit the mall than males'''

#Check Age distribution
mallcustomers[['Age']].plot(kind='hist',bins=[0,10,20,30,40,50,60,70,80,90,100],rwidth=0.9)
mtp.show()
'''most customers are in the age of 20-40'''

#Spending Habits 
mallcustomers.groupby('Gender').mean()
''' Mean spending amount of women is 51.526786
    Mean spending amount of men is 48.511364'''


#Annual Income Data Analysis
print("Mean of Annual Income (k$) of Female:",female['Annual Income (k$)'].mean())
print("Mean of Annual Income (k$) of Male:",male['Annual Income (k$)'].mean())
'''Mean of Annual Income (k$) of Female: 59.25
Mean of Annual Income (k$) of Male: 62.22727272727273
Insights-
    More women visit the mall than men.
    Women's average Salary is lesser than men.
    Women spend more than men.'''
    

# Finding Correlations
sns.pairplot(data=mallcustomers)

mtp.figure(figsize=(10,5))
sns.heatmap(mallcustomers.corr(),annot=True,cmap='hsv',fmt='.2f',linewidths=2)
mtp.show()

mtp.figure(figsize=(14,5))
mtp.subplot(1,3,1)
sns.distplot(mallcustomers['Age'])
mtp.title('Distplot of Age')
mtp.subplot(1,3,2)
sns.distplot(mallcustomers['Spending Score (1-100)'],hist=False)
mtp.title('Distplot of Spending Score (1-100)')
mtp.subplot(1,3,3)
sns.distplot(mallcustomers['Annual Income (k$)'])
mtp.title('Annual Income (k$)')
mtp.show()

#K-means clustering : Chose k=5
'''For feature Selection I picked only Annual Income and Spending score to create clusters unbiased with Gender and Age factors'''
X= mallcustomers.iloc[:, [3,4]].values
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

#Visualizing Clusters
mtp.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
mtp.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
mtp.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
mtp.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
mtp.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
mtp.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
mtp.title('Clusters of customers')
mtp.xlabel('Annual Income (k$)')
mtp.ylabel('Spending Score (1-100)')
mtp.legend()
mtp.show()
