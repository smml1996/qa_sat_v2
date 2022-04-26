from math import exp
from copy import deepcopy

from scipy import asarray

from utils import evaluate_cnf_formula

from scipy import rand, randn
import random


def simulated_annealing(bqm, sample, n_iterations, step_size, temp, or_gates, seed=1):
	random.seed(seed)
	bounds = asarray([[.0, 1.0]])
	# evaluate the initial point
	best_eval = evaluate_cnf_formula(sample, or_gates, bqm)
	# current working solution
	curr, curr_eval = sample, best_eval
	# run the algorithm
	for i in range(n_iterations):
		# take a step
		candidate = deepcopy(curr)
		for key in candidate.keys():
			candidate[key] = round(candidate[key] + randn(len(bounds))[0] * step_size,0) % 2
		# candidate = curr + randn(len(bounds)) * step_size
		# evaluate candidate point
		candidate_eval = evaluate_cnf_formula(candidate, or_gates, bqm)
		# check for new best solution
		if candidate_eval < best_eval:
			# store new best point
			sample, best_eval = candidate, candidate_eval
			# report progress
			print('>%d f(%s) = %.5f' % (i, sample, best_eval))
		# difference between candidate and current point evaluation
		diff = candidate_eval - curr_eval
		# calculate temperature for current epoch
		t = temp / float(i + 1)
		# calculate metropolis acceptance criterion
		metropolis = exp(-diff / t)
		# check if we should keep the new point
		if diff < 0 or rand() < metropolis:
			# store the new current point
			curr, curr_eval = candidate, candidate_eval
		if candidate_eval == 0:
			break
	return [sample, best_eval]