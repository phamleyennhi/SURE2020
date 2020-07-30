'''
SURE Program 2020
Student: Nhi Pham
Advisor: Esteban Tabak
'''
import matplotlib.pyplot as plt
from hmmlearn import hmm
import numpy as np
import math

#  hidden state z, observation x, initial probability pi, and distribution parameter Theta (Gaussian)
z = ('Sunny', 'Rainy')
x = ('Happy', 'Sad')
pi = {'Sunny': 0.5, 'Rainy': 0.5}
#  transition probability matrix A
A = {'Sunny' : {'Sunny': 0.8, 'Rainy': 0.2},
   'Rainy' : {'Sunny': 0.4, 'Rainy': 0.6},}
#  emission probability matrix B
B = {'Sunny' : {'Happy': 0.7, 'Sad': 0.3},
   'Rainy' : {'Happy': 0.1, 'Sad': 0.9},}

# Build the HMM model
model = hmm.GaussianHMM(n_components=2, covariance_type="full")
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.8, 0.2],
                            [0.4, 0.6]])
model.means_ = np.array([[0.0, 0.0], 
						 [-2.0, 2.0]])
model.covars_ = np.tile(np.identity(2), (2, 1, 1))

# Generate samples (observation, hidden state) = (x_i,z_i) 
X, Z = model.sample(100)

# Plot the sampled data
plt.plot(X[:, 0], X[:, 1], ".-", label="observations",
         mfc="black")
for i, m in enumerate(model.means_):
    plt.text(m[0], m[1], 'Component %i' % (i + 1),
             size=15, horizontalalignment='center',
             bbox=dict(alpha=0.8))
plt.legend(loc='best')
# plt.show()



'''
The HHM can be characterized by THREE fundamental problems:
    Problem 1 (Likelihood):  What is the probability of an observed sequence?
    Problem 2 (Decoding): What is the most likely series of states to generate the observations?
    Problem 3 (Learning): How can we learn values for the HMM's parameters A given some data?
'''

# Problem 1: give the observed sequence X, calculate the probability?
print(math.exp(model.score(X)))

# Problem 2 (Decoding): What is the most likely series of states to generate the observations?
print(model.decode(X, algorithm='viterbi'))
print(model.predict(X))

# Problem 3 (Learning): How can we learn values for the HMM's parameters A given some data?
# Training HMM parameters and inferring the hidden states
newModel = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
newModel.fit(X)
newZ = newModel.predict(X)
print(newZ)
print(newModel.transmat_)

