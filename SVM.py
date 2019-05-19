#1001553524
#Varsha Rani Chawan
#support vector machine

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

X = np.array([[1, 2],[2, 3],[2, 1],[3, 4],[1, 3],[4, 4]])
Y = [-1,1,-1,1,-1,1]

clf = svm.SVC(kernel='linear', C = 2)
clf.fit(X,Y)


support_vectors = clf.support_vectors_
print("========================================================================")
print("Support Vectors are :",support_vectors )
print("========================================================================")

# plot decision boundary
fig = plt.figure()
ax = fig.gca()
ax.set_xlim([0,5])
ax.set_ylim([0,5])
for x ,y in zip(X,Y):
    ax.plot(x[0],x[1] ,'g+' if (y == 1) else  'ro' , markersize=7)
x1,x2 = np.meshgrid(np.arange(0.0,5.0,0.01),np.arange(0.0,5.0,0.01))
z = clf.predict(np.c_[x1.ravel(),x2.ravel()])
z = z.reshape(x1.shape)
plt.contour(x1,x2,z)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
