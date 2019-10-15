import matplotlib.pyplot as plt

#import load_iris function from datasets module provided by sklearn
from sklearn.datasets import load_iris

import numpy as np



#variable fig referencing the scatterplot diagram to be used
fig = plt.figure()


# plot the figures.
# Code represents the  number of rows in the scatterplot diagram (the number of subplots below one another),
# the number of colunms in the scatterplot (the number of subplots next to each other)
# and the last digit indicates the figure/subplot number being plotted 
fig1 = plt.subplot(4, 3, 1)
fig2 = plt.subplot(4, 3, 2)
fig3 = plt.subplot(4, 3, 3)
fig4 = plt.subplot(4, 3, 4)
fig5 = plt.subplot(4, 3, 5)
fig6 = plt.subplot(4, 3, 6)
fig7 = plt.subplot(4, 3, 7)
fig8 = plt.subplot(4, 3, 8)
fig9 = plt.subplot(4, 3, 9)
fig10 = plt.subplot(4, 3, 10)
fig11= plt.subplot(4, 3, 11)
fig12 = plt.subplot(4, 3, 12)

#save "bunch" object containing iris dataset and its attributes
#figure and subplot created
iris = load_iris()
data = np.array(iris['data'])
targets = np.array(iris['target'])#target represents what we will be predicting, ie. 0, 1 0r 2 (the 3 classes)

#a dictionary is created to reference colors for the 3 classes
cd = {0:'r', 1:'g', 2:'b'}
cols = np.array([cd[target] for target in targets])

# the scatterplot for each subplot
fig1.scatter(data[:,0], data[:,1], c=cols)
fig2.scatter(data[:,0], data[:,2], c=cols)
fig3.scatter(data[:,0], data[:,3], c=cols)
fig4.scatter(data[:,1], data[:,0], c=cols)
fig5.scatter(data[:,1], data[:,2], c=cols)
fig6.scatter(data[:,1], data[:,3], c=cols)
fig7.scatter(data[:,2], data[:,0], c=cols)
fig8.scatter(data[:,2], data[:,1], c=cols)
fig9.scatter(data[:,2], data[:,3], c=cols)
fig10.scatter(data[:,3], data[:,0], c=cols)
fig11.scatter(data[:,3], data[:,1], c=cols)
fig12.scatter(data[:,3], data[:,2], c=cols)

#save plot
plt.savefig('Iris.png')
#display the plot created
plt.show()
