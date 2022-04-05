# Code for the experiments

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats

import csv

CSV_DIR = 'data/'

NUM_CHOICES = 10 # ratings in deciles

# Plot aesthetics
fontsize = 25
legendsize = 20
ticksize = 17.5
linewidth = 2.5
markersize = 10
markeredgewidth = 4
axissize = 17.5
MARKERS = ['o','s','<', 'd', '*']
LINESTYLES = ['dotted', 'dashed', 'dashdot', 'solid', (0, (1, 5))]
COLORS = ['C0', 'C1', 'C2', 'C3', 'C4']

# Read data from CSV
def read_csv(csv_file):

	## Read from all workers
	with open(csv_file, 'r') as f:
		file = csv.DictReader(f)
		workers = []

		for row in file:
			workers.append(dict(row))

	## Format numeric
	for w in workers:
		# int/float
		for k, v in w.items():
			try:
				w[k] = int(v)
				continue
			except ValueError: pass

			try:
				w[k] = float(v)
			except ValueError: pass

	W = len(workers)

	print('#workers: %d' % W)
	return workers

# Parse data
# Q: number of questions, or equivalently number of items (5 or 10)
def parse_data(workers, Q):
	W = len(workers)

	# parse answers
	answers = np.zeros((W, Q))
	for iw in range(W):
		for q in range(Q):
			answers[iw, q] = workers[iw]['score%d' % (q+1)]
	
	# parse order
	values = np.zeros((W, Q))
	orders = np.zeros((W, Q), dtype=int)
	for iw in range(W):
		for q in range(Q):
			values[iw, q] = workers[iw]['size%d' % (q+1)]

		orders[iw, :] = scipy.stats.rankdata(values[iw, :]) # 1-idxed
			
	return answers, orders

# Parse comparison from the conflict experiment (the comaprison between the first and last items)
def parse_data_compare(workers):
	W = len(workers)
	comparisons = np.zeros(W, dtype=int)

	for iw in range(W):
		comparisons[iw] = workers[iw]['comparison']

	return comparisons

# Existence experiment:
# Plot the parameters of the model
def analyze_existence(answers, orders):
	orders = orders - 1 # 1-idx to 0-idx
	(W, Q) = answers.shape
	
	# SCORES (Q x Q): each cell (r, t) means the score received by item (of absolute rank r) at position t
	scores = np.empty((Q, Q), dtype=object)
	for i in range(Q):
		for t in range(Q):
			scores[i, t] = np.array([])

	for iw in range(W):
		for t in range(Q):
			r = orders[iw, t]
			scores[r, t] = np.append(scores[r, t], answers[iw, t])


	(fig, ax) = plt.subplots()
	# Compute (mean, std err of mean, count) for each cell of SCORE
	means = np.zeros((Q, Q))
	stds = np.zeros((Q, Q))
	counts = np.zeros((Q, Q), dtype=int)

	for i in range(Q):
		for j in range(Q):
			counts[i, j] = len(scores[i, j])
			means[i, j] = np.mean(scores[i, j])
			stds[i, j] = np.std(scores[i, j]) / np.sqrt(len(scores[i, j]))

	# 240 participants
	assert(np.all(counts == 48))
	
	for i in np.arange(Q-1, -1, -1):
		ax.errorbar(np.arange(Q)+1, means[i, :], yerr=stds[i, :], label='Rank %d' % (i+1),
				marker=MARKERS[i], linestyle=LINESTYLES[i], color=COLORS[i], 
				markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth)

	ax.tick_params(axis='x', labelsize=ticksize)
	ax.tick_params(axis='y', labelsize=ticksize)

	ax.set_xticks(np.arange(Q)+1)
	ax.set_xlabel('Position (%s)' % r'$t$', fontsize=axissize)
	ax.set_ylabel('Reported score', fontsize=axissize)
	ax.set_ylim([0.5, NUM_CHOICES])
	ax.legend(fontsize=legendsize, loc='center left', bbox_to_anchor=(1.0, 0.5))
	plt.tight_layout()
	plt.show()


# Conflict experiment
# 	Group 1: [1,3,4,5,2]
# 	Group 2: [2,3,4,5,1]
# 	COMPARISONS: 1 if the participant reports the first item is larger
#				 2 if the participant reports the last item is larger
def analyze_conflict(answers, orders, comparisons, pair=(0, 4)):
	W = answers.shape[0]
	(p1, p2) = pair

	# group to {conflict, no conflict}
	has_conflict = np.zeros(W, dtype=bool)
	for iw in range(W):
		order = orders[iw, :]

		if np.array_equal(orders[iw, :],  [1, 3, 4, 5, 2]):
			has_conflict[iw] = True
		else:
			assert(np.array_equal(orders[iw, :], [2, 3, 4, 5, 1]))

	select_rating_correct = np.zeros(W, dtype=bool) # comparison induced by reported socres
	select_ranking_correct = np.zeros(W, dtype=bool) # comparison reported

	for iw in range(W):
		if orders[iw, p1] > orders[iw, p2]: # first item > last item
			select_rating_correct[iw] = answers[iw, p1] > answers[iw, p2]
			select_ranking_correct[iw] = (comparisons[iw] == 1) # first is greater
		else: # first item < last item
			select_rating_correct[iw] = answers[iw, p1] < answers[iw, p2]
			select_ranking_correct[iw] = (comparisons[iw] == 2) # last is greater

		# TODO: handle ties
		# select_rating_correct[iw] += (answers[iw, p1] == answers[iw, p2])

	for mode in ['conflict', 'no_conflict']:
		if mode == 'conflict':
			select = has_conflict
		else: # no conflict
			select = np.logical_not(has_conflict)

		assert(np.sum(select) == 50) # 100 participants in total
		print('====== mode ======')
		print(mode)

		print('  correct rating + correct ranking: %d' % \
				np.sum(np.logical_and(select_rating_correct[select], select_ranking_correct[select])))
		print('  correct rating + wrong ranking: %d' % \
				np.sum(np.logical_and(select_rating_correct[select], 
									  np.logical_not(select_ranking_correct[select]))))
		print('  wrong rating + correct ranking: %d' % \
				np.sum(np.logical_and(np.logical_not(select_rating_correct[select]), 
									  select_ranking_correct[select])))
		print('  wrong rating + wrong ranking: %d' % \
				np.sum(np.logical_and(np.logical_not(select_rating_correct[select]), 
									  np.logical_not(select_ranking_correct[select]))))

# Analyze relative nature of the scores
def analyze_relativity(answers, orders):
	(W, Q) = answers.shape
	Q_half = int(Q/2)

	data_lh_last = np.array([]) # last half in L-H group
	data_hh_last = np.array([]) # last half in H-H group
	data_hh_first = np.array([]) # first half in H-H group
	for iw in range(W):
		answer = answers[iw, :]
		order = orders[iw, :]
		if len(np.unique(order)) == 5: # heuristic for (H, H)
			data_hh_first = np.concatenate((data_hh_first, answer[:Q_half]))
			data_hh_last = np.concatenate((data_hh_last, answer[Q_half:]))
		else: # (L, H)
			assert(len(np.unique(order)) == 10)
			data_lh_last = np.concatenate((data_lh_last, answer[Q_half:]))

	n = 25 * Q_half # 25 participants per group
	assert(len(data_hh_last) == n) 
	assert(len(data_lh_last) == n)

	# Compare the last half across two groups
	print('H-H last half: %.2f +/- %.2f' % (np.mean(data_hh_last), np.std(data_hh_last) / np.sqrt(n)))
	print('L-H last half: %.2f +/- %.2f' % (np.mean(data_lh_last), np.std(data_lh_last) / np.sqrt(n)))
	print('p-value (one-side): %.5f' % perm_test(data_lh_last, data_hh_last))

	# Compare {first half, last half} within the H-H group
	assert(len(data_hh_first) == n)
	print('H-H first half: %.2f +/- %.2f' % (np.mean(data_hh_first), np.std(data_hh_first) / np.sqrt(n)))
	print('H-H last half: %.2f +/- %.2f' % (np.mean(data_hh_last), np.std(data_hh_last) / np.sqrt(n)))	
	print('p-value (one-side): %.5f' % perm_test(data_hh_first, data_hh_last))

# Perform permutation test (unpaired, one-sided)
# 	H0: ys1 == ys2
# 	H1: ys1 > ys2
# Return: p-value
def perm_test(ys1, ys2, repeat=100000):
	n1 = len(ys1)
	n2 = len(ys2)
	diff_test = np.mean(ys1) - np.mean(ys2)

	ys = np.concatenate((ys1, ys2))
	count = 0
	for r in range(repeat):
		perm = np.random.permutation(n1 + n2)
		ys_permuted = ys[perm]

		diff = np.mean(ys_permuted[:n1]) - np.mean(ys_permuted[n1:])
		if diff > diff_test:
			count += 1

	return count / repeat

if __name__ == '__main__':
	np.random.seed(0)

	# existence experiment
	print('====== Existence experiment ======')
	Q = 5 # number of items
	CSV_FILE_EXISTENCE = CSV_DIR + 'existence.csv'
	workers_existence = read_csv(csv_file=CSV_FILE_EXISTENCE)	
	answers_existence, orders_existence = parse_data(workers_existence, Q)
	analyze_existence(answers_existence, orders_existence)
	print()

	# relative experiment
	print('====== Relative experiment ======')
	Q = 10
	CSV_FILE_RELATIVE = CSV_DIR + 'relativity.csv'
	workers_relative = read_csv(csv_file=CSV_FILE_RELATIVE)
	answers_relative, orders_relative = parse_data(workers_relative, Q)
	analyze_relativity(answers_relative, orders_relative)
	print()

	# conflict experiment
	print('====== Conflict experiment ======')
	Q = 5
	CSV_FILE_COMPARE = CSV_DIR + 'conflict.csv'
	workers_compare = read_csv(csv_file=CSV_FILE_COMPARE)
	answers_compare, orders_compare = parse_data(workers_compare, Q)
	comparisons = parse_data_compare(workers_compare)
	analyze_conflict(answers_compare, orders_compare, comparisons)
	print()
