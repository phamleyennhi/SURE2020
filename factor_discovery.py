# import packages
import numpy as np
import matplotlib.pyplot as plt
import random
import math

z0 = 0
def generate_data(mu, sigma, gamma, N, useNorm=True):
	# mu, sigma = 0, 1 # mean and standard deviation
	# N = 100000
	# gamma = 0.1
	# delta = np.random.normal(mu, 1, N)
	delta = np.random.normal(mu, sigma, N)
	z_vec = []
	z_vec.append(gamma*z0 + delta[0])
	for i in range(1, N):
	    z_vec.append(gamma*z_vec[i-1] + delta[i])  
	
	x_vec = []
	for i in range(N):
		if useNorm:
			# x_i = np.random.normal(z_vec[i], sigma)
			x_i = np.random.normal(z_vec[i], 1)
		else:
			x_i = z_vec[i]
		x_vec.append(x_i)

	# convert to np array
	x_vec = np.array(x_vec)
	z_vec = np.array(z_vec)
	return x_vec, z_vec


def step_one(lmda, gamma, x_vec, N, y_vec):
	z = []
	for i in range(N-1):
		if y_vec is None:
			z_i = (2.0*x_vec[i])/(2.0+2.0*(gamma**2)*lmda)
		else:
			z_i = (2.0*lmda*gamma*y_vec[i+1] + 2.0*x_vec[i])/(2.0+2.0*(gamma**2)*lmda)
		z.append(z_i)
	z.append(x_vec[N-1])
	z = np.array(z, dtype='float')
	y_vec = np.array(y_vec, dtype='float')
    
	MSE = None
	if y_vec is not None:
		MSE = np.sum((y_vec - z)**2)/N
	return z, MSE


def step_two(z, N):
	Z = []
	Z.append(z0)
	for i in range(N-1):
		Z.append(z[i])
	Z = np.matrix(Z, dtype='float')
	ZZ = np.matmul(Z, Z.transpose())
	ZZ = np.linalg.pinv(ZZ)
	ZZ_vec = np.matmul(Z, z.transpose())
	gamma_vec = ZZ*ZZ_vec
	gamma = gamma_vec.item(0)
	return gamma



def repeat_until_convergence(lmda, x_vec, z_vec, threshold, N):
	gamma = random.randrange(0,1)
	iteration = 0
	while True:
		if iteration == 0:
			y_vec = None
		else:
			y_vec = found_z
		found_z, MSE = step_one(lmda, gamma, x_vec, N, y_vec)
		gamma = step_two(found_z, N)
		iteration += 1
		# print(MSE)
		if MSE != None and MSE < threshold:
			# print("When lmda is", lmda, ", the found gamma is:", gamma)
			break
		# no convergence
		if iteration > 1000:
			print("No convergence, lmda =", lmda)
			return "nan"
	return gamma


def main():
	global_gammas = {}
	best_lambdas = {}
	mu = 0
	all_sigmas = [0.25, 0.5, 0.75, 1, 2, 4, 8, 16]
	# all_Ns = [1000, 10000, 100000]
	all_Ns = [1000]
	all_lambdas = [0,0.001,0.005,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,\
				   1.6,1.7,1.8,1.9,2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
	# all_lambdas = [0.5]
	all_gammas = [0.3, 0.5, 0.7, 0.9]
	
	for N in all_Ns:
		print(N)
		global_gammas[str(N)] = {}
		best_lambdas[str(N)] = {}
		for gamma in all_gammas:
			global_gammas[str(N)][str(gamma)] = {}
			best_lambdas[str(N)][str(gamma)] = {}
			for sigma in all_sigmas:
				global_gammas[str(N)][str(gamma)][str(sigma)] = []
				# global_gammas[str(N)][str(gamma)][str(sigma)] = 0
				x_vec, z_vec = generate_data(mu, sigma, gamma, N, useNorm=True)
				found_gammas = []
				best_lambda = -1
				best_gamma = -1
				for lmda in all_lambdas:
					threshold = 0.000000001
					found_gamma = repeat_until_convergence(lmda, x_vec, z_vec, threshold, N)
					found_gammas.append(found_gamma)
					if found_gamma == "nan":
						continue
					else:
						if best_gamma == -1:
							best_lambda = lmda
							best_gamma = found_gamma
						elif abs(gamma - best_gamma) > abs(gamma - found_gamma):
							best_lambda = lmda
							best_gamma = found_gamma
				best_lambdas[str(N)][str(gamma)][str(sigma)] = best_lambda
				global_gammas[str(N)][str(gamma)][str(sigma)] = found_gammas

	# for i in range(len(global_lambdas)):
	#     plt.plot(global_lambdas[i], global_gammas[i], '.')

	# dt = ["100", "1000", "10000", "100000"]
	# plt.legend(dt)
	print("GAMMAS")
	print(global_gammas)
	print("BEST LAMDBAS")
	print(best_lambdas)

	for gamma in all_gammas:
		for sigma in all_sigmas:
			for N in all_Ns:
				print("N, gamma, sigma, best lambda: " + str(N) + ", " + str(gamma) + ", " + str(sigma) + \
					  ", " + str(best_lambdas[str(N)][str(gamma)][str(sigma)]))
				print()
			print()


def experiment(N, mu, sigma, gamma, threshold = 0.000000001):
	lmda=(sigma**0.5)*0.5
	x_vec, z_vec = generate_data(mu, sigma, gamma, N)
	found_gamma = repeat_until_convergence(lmda, x_vec, z_vec, threshold, N)
	print("Real gamma, Found gamma:", gamma, ",", found_gamma)

# for i in range (10):
# 	experiment(10000, 0, 8, 0.3)

main()



