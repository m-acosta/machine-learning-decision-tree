#-------------------------------------------------------------------------
# AUTHOR: Michael Acosta
# FILENAME: decision_tree.py
# SPECIFICATION: Create a decision tree from a set of features.
# FOR: CS 4210- Assignment #1
# TIME SPENT: About 1.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
Age = {
   "Young": 1,
   "Presbyopic": 2,
   "Prepresbyopic": 3,
}
SpectaclePrescription = {
   "Myope": 1,
   "Hypermetrope": 2,
}
Astigmatism = {
   "No": 1,
   "Yes": 2,
}
TearProductionRate = {
   "Reduced": 1,
   "Normal": 2,
}
X = db.copy() #make a copy of file read in
for i, row in enumerate(db): 
  #transform features based on dictionary
  X[i] = [Age[row[0]], SpectaclePrescription[row[1]], 
          Astigmatism[row[2]], TearProductionRate[row[3]]]

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
RecommendedLenses = {
   "No": 1,
   "Yes": 2,
}
Y = [0] * len(db) #create empty list
for i, row in enumerate(db):
   Y[i] = RecommendedLenses[row[4]] #transform label based on dictionary

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()