'''
SURE Program 2020
Student: Nhi Pham
Advisor: Esteban Tabak

Likelihood Computation: Forward Algorithm
'''
import math
import random

def likelihood(X, A, B, pi):

	#  initialize alpha
	alpha = []
	for i in range(len(z)):
		tmp = []
		for j in range(len(X) + 1):
			tmp.append(0.0)
		alpha.append(tmp)

	#  bottom-up approach 
	for t in range (len(X) + 1):

		if t == 0:
			for i in range(len(z)):
				alpha[i][0]= pi[i]

		else:
			current_observation = X[i-1]
			# cell alpha[j][t] being in the state s_j at time t
			for j in range (len(z)): 
				alpha[j][t] = 0.0
				#  summing over the probabilities of every path that could lead to this cell alpha[j][t] 
				for i in range (len(z)):
					# print("j, t: ", j, t)
					# print(alpha[i][t-1], A[i][j], B[j][current_observation])
					alpha[j][t] += alpha[i][t-1]*A[i][j]*B[j][current_observation]


	# likelihood computation
	likelihood_probability = 0.0
	for state in range(len(z)):
		likelihood_probability += alpha[state][len(X)-1]


	return likelihood_probability




#  hidden state z, observation x, initial probability pi, transition matrix A, emission matrix B
z = ['Sunny', 'Rainy']
x = ['Happy', 'Sad']
pi = [0.5, 0.5]  # (0: sunny, 1: rainy)

#  transition probability matrix A
A = [[0.8, 0.2], 
	[0.4, 0.6]] 
#  emission probability matrix B
B = [[0.7, 0.3], 
	[0.1, 0.9]] 


# Calculate the probability of a sequence of observations 
X = []
for i in range (3):
	X.append(random.randint(0,1))  # (0: happy, 1: sad)

print("Sequence of observations", X)
print("The likelihood probability of the given sequence of observations is", likelihood(X, A, B, pi))



