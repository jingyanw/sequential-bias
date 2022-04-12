# Run simulation (Figure 4 and Figure 5)

import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import scipy
import scipy.stats
import time

COLORS = ['C0', 'C1']
fontsize = 25
legendsize = 20
ticksize = 17.5
linewidth = 2.5
markersize = 10
markeredgewidth = 4
axissize = 17.5
MARKERS = ['o','s']
LINESTYLES = ['solid', (0, (1, 1))]
LABELS = ['Least squares', 'Induced ranking']

# ============
# Estimator
# ============
# Least-squares estimator implemented by the insertion algorithm
def estimate_order_insertion(scores):
	n = len(scores)
	order = np.array([])

	for i in range(0, n): # 1 = second item (0-idxed)
		thresholds = np.arange(1, i+2) / (i+2)
		pos = np.argmin(np.abs(scores[i] - thresholds)) # for simplicity, take the smaller position in tie-breaking
		order = np.insert(order, pos, i)

	return np.argsort(order)

# Compute noiseless scores according to the parametric model
# Returns: 
# 	SCORES: n-array
def compute_scores(xs):

	# Compute the noiseless score of the last item
	def compute_scores_last(xs):
		n = len(xs)
		x = xs[-1]
		rank = np.sum(x > xs[:-1]) + np.sum(x == xs[:-1]) *0.5 +1 # 1-indexed rank of last element
		return int(rank) / (n+1)

	n = len(xs)
	scores = np.zeros(n)
	for i in range(n):
		scores[i] = compute_scores_last(xs[:i+1])
	return scores

# ============
# Dist
# ============
def dist_entrywise(perm, perm_gt):
	return np.abs(scipy.stats.rankdata(perm) - scipy.stats.rankdata(perm_gt))

# Spearmans' footrule (normalized)
def dist_footrule(perm, perm_gt):
	n = len(perm)
	return np.mean(dist_entrywise(perm, perm_gt)) / n

# ============
# Simulation
# ============
# Plot the Spearman's footrule and maximal entrywise error (Figure 4)
def simulate_err():
	## Vary n
	delta = 0.1 # noise level
	ns = np.array([20, 50, 100, 250, 500])
	N = len(ns)
	repeat = 1000

	errs_insert_sf = np.zeros((N, repeat))
	errs_induce_sf = np.zeros((N, repeat))

	errs_insert_max_expect = np.zeros((N, 2)) # (mean, std)
	errs_induce_max_expect = np.zeros((N, 2)) # (mean, std)

	idxs_insert = np.zeros(N)
	idxs_induce = np.zeros(N)

	tic = time.time()
	print('===Vary n===')
	for i in range(N):
		print('n: %d/%d (%.1f sec)' % (i+1, N, time.time() - tic))
		n = ns[i]

		errs_insert = np.zeros((n, repeat)) # entrywise
		errs_induce = np.zeros((n, repeat))

		for r in range(repeat):
			perm = np.random.permutation(n)
			scores = compute_scores(perm)
			noise_unit = np.random.uniform(low=-1, high=1, size=n)

			ys = scores + noise_unit * delta
			est_insert = estimate_order_insertion(ys)
			est_induce = scipy.stats.rankdata(ys)

			errs_insert_sf[i, r] = dist_footrule(perm, est_insert)
			errs_induce_sf[i, r] = dist_footrule(perm, est_induce)
			errs_insert[:, r] = dist_entrywise(perm, est_insert)
			errs_induce[:, r] = dist_entrywise(perm, est_induce)

		# aggregate max error
		means_insert, stds_insert = np.mean(errs_insert, axis=1), np.std(errs_insert, axis=1)
		idx_insert = np.argmax(means_insert)
		errs_insert_max_expect[i, :] = means_insert[idx_insert] / n, stds_insert[idx_insert] / n

		means_induce, stds_induce = np.mean(errs_induce, axis=1), np.std(errs_induce, axis=1)
		idx_induce = np.argmax(means_induce)
		errs_induce_max_expect[i, :] = means_induce[idx_induce] / n, stds_induce[idx_induce] / n

		idxs_insert[i] = idx_insert
		idxs_induce[i] = idx_induce


	## Plot
	# Spearman's footrule
	fig, ax = plt.subplots()
	ax.errorbar(ns, np.mean(errs_insert_sf, axis=1), yerr=np.std(errs_insert_sf, axis=1) / np.sqrt(repeat), label=LABELS[0],
				marker=MARKERS[0], linestyle=LINESTYLES[0], color=COLORS[0],
				markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth)
	ax.errorbar(ns, np.mean(errs_induce_sf, axis=1), yerr=np.std(errs_induce_sf, axis=1) / np.sqrt(repeat), label=LABELS[1],
				marker=MARKERS[1], linestyle=LINESTYLES[1], color=COLORS[1],
				markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth)
	ax.set_xlabel('Number of item (%s)' % r'$n$', fontsize=axissize)
	ax.set_ylabel("Spearman's footrule distance", fontsize=axissize)

	ax.tick_params(axis='x', labelsize=ticksize)
	ax.tick_params(axis='y', labelsize=ticksize)
	ax.set_xticks([20, 50, 100, 250, 500])
	ax.set_xticklabels([r'$20$', r'$50$', r'$100$', r'$250$', r'$500$'])

	ax.set_ylim([0, None])
	ax.legend(fontsize=legendsize)
	fig.tight_layout()

	# Maximal entrywise error
	fig, ax = plt.subplots()
	ax.errorbar(ns, errs_insert_max_expect[:, 0], yerr=errs_insert_max_expect[:, 1] / np.sqrt(repeat), label='insert',
				marker=MARKERS[0], linestyle=LINESTYLES[0], color=COLORS[0], 
				markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth)
	ax.errorbar(ns, errs_induce_max_expect[:, 0], yerr=errs_induce_max_expect[:, 1] / np.sqrt(repeat), label='induce',
				marker=MARKERS[1], linestyle=LINESTYLES[1], color=COLORS[1],
				markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth)
	ax.set_xlabel('Number of item (%s)' % r'$n$', fontsize=axissize)
	ax.set_ylabel('Maximum entry-wise error', fontsize=axissize)

	ax.tick_params(axis='x', labelsize=ticksize)
	ax.tick_params(axis='y', labelsize=ticksize)
	ax.set_xticks([20, 50, 100, 250, 500])
	ax.set_xticklabels([r'$20$', r'$50$', r'$100$', r'$250$', r'$500$'])

	ax.set_ylim([0, None])
	fig.tight_layout()

	plt.show()

	######
	## Vary delta
	deltas = [0.02, 0.05, 0.1, 0.2]
	D = len(deltas)
	n = 1000
	repeat = 1000

	errs_insert_sf = np.zeros((D, repeat))
	errs_induce_sf = np.zeros((D, repeat))
	errs_insert_entrywise = np.zeros((D, n, repeat))
	errs_induce_entrywise = np.zeros((D, n, repeat))

	idxs_insert = np.zeros(D)
	idxs_induce = np.zeros(D)

	print('===Vary delta===')
	for idd in range(D):
		print('delta: %d/%d (%.1f sec)' % (idd+1, D, time.time() - tic))
		delta = deltas[idd]

		for r in range(repeat):
			perm = np.random.permutation(n)
			scores = compute_scores(perm)
			noise_unit = np.random.uniform(low=-1, high=1, size=n)

			ys = scores + noise_unit * delta
			est_insert = estimate_order_insertion(ys)
			est_induce = scipy.stats.rankdata(ys)

			errs_insert_sf[idd, r] = dist_footrule(perm, est_insert)
			errs_induce_sf[idd, r] = dist_footrule(perm, est_induce)
			errs_insert_entrywise[idd, :, r] = dist_entrywise(perm, est_insert)
			errs_induce_entrywise[idd, :, r] = dist_entrywise(perm, est_induce)

	## Plot
	# Spearman's footrule
	fig, ax = plt.subplots()
	ax.errorbar(deltas, np.mean(errs_insert_sf, axis=1), yerr=np.std(errs_insert_sf, axis=1) / np.sqrt(repeat), label='insert',
				marker=MARKERS[0], linestyle=LINESTYLES[0], color=COLORS[0],
				markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth)
	ax.errorbar(deltas, np.mean(errs_induce_sf, axis=1), yerr=np.std(errs_induce_sf, axis=1) / np.sqrt(repeat), label='induce',
				marker=MARKERS[1], linestyle=LINESTYLES[1], color=COLORS[1],
				markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth)
	ax.set_xlabel('Noise level (%s)' % r'$\delta$', fontsize=axissize)
	ax.set_ylabel("Spearman's footrule distance", fontsize=axissize)

	ax.tick_params(axis='x', labelsize=ticksize)
	ax.tick_params(axis='y', labelsize=ticksize)
	ax.set_xticks([0.02, 0.05, 0.1, 0.2])
	ax.set_xticklabels([r'$0.02$', r'$0.05$', r'$0.1$', r'$0.2$'])

	ax.set_ylim([0, None])
	fig.tight_layout()

	# Maximal entrywise error
	fig, ax = plt.subplots()
	means_insert = np.mean(errs_insert_entrywise, axis=2) # D x n
	means_induce = np.mean(errs_induce_entrywise, axis=2)

	errs_insert = np.zeros((D, repeat))
	errs_induce = np.zeros((D, repeat))
	for idd in range(D):
		idx_insert = np.argmax(means_insert[idd, :])
		idx_induce = np.argmax(means_induce[idd, :])

		idxs_insert[idd] = idx_insert
		idxs_induce[idd] = idx_induce

		errs_insert[idd, :] = errs_insert_entrywise[idd, idx_insert, :] / n
		errs_induce[idd, :] = errs_induce_entrywise[idd, idx_induce, :] / n

	ax.errorbar(deltas, np.mean(errs_insert, axis=1), yerr=np.std(errs_insert, axis=1) / np.sqrt(repeat), label=LABELS[0],
				marker=MARKERS[0], linestyle=LINESTYLES[0], color=COLORS[0],
				markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth)
	ax.errorbar(deltas, np.mean(errs_induce, axis=1), yerr=np.std(errs_induce, axis=1) / np.sqrt(repeat), label=LABELS[1],
				marker=MARKERS[1], linestyle=LINESTYLES[1], color=COLORS[1],
				markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth)

	(handles, labels) = ax.get_legend_handles_labels()
	ax.set_xlabel('Noise level (%s)' % r'$\delta$', fontsize=axissize)
	ax.set_ylabel('Maximum entry-wise error', fontsize=axissize)

	ax.tick_params(axis='x', labelsize=ticksize)
	ax.tick_params(axis='y', labelsize=ticksize)
	ax.set_xticks([0.02, 0.05, 0.1, 0.2])
	ax.set_xticklabels([r'$0.02$', r'$0.05$', r'$0.1$', r'$0.2$'])

	ax.set_ylim([0, None])
	fig.tight_layout()

	plt.show()

# Plot error at each individual position (Figure 5)
def simulate_err_per_item():
	delta = 0.1
	n = 100
	repeat = 1000

	errs_induce = np.zeros((n, repeat))
	errs_insert = np.zeros((n, repeat))

	tic = time.time()
	for r in range(repeat):
		perm = np.random.permutation(n)
		scores = compute_scores(perm)

		noise_unit = np.random.uniform(low=-1, high=1, size=n)
		ys = scores + noise_unit * delta

		est_insert = estimate_order_insertion(ys)
		est_induce = scipy.stats.rankdata(ys)

		errs_insert[:, r] = np.abs(est_insert - perm) / n
		errs_induce[:, r] = np.abs(est_induce - perm) / n

	# Plot error per item
	fig, ax = plt.subplots()
	ax.errorbar(np.arange(n)+1, np.mean(errs_insert, axis=1),
				linestyle=LINESTYLES[0], linewidth=linewidth,
				color=COLORS[0], label=LABELS[0])
	ax.errorbar(np.arange(n)+1, np.mean(errs_induce, axis=1),
				linestyle=LINESTYLES[1], color=COLORS[1], linewidth=linewidth,
				label=LABELS[1])

	ax.set_xlabel('Position (%s)' % r'$t$', fontsize=axissize)
	ax.set_ylabel('Entry-wise error', fontsize=axissize)

	ax.tick_params(axis='x', labelsize=ticksize)
	ax.tick_params(axis='y', labelsize=ticksize)

	ax.set_xlim(0, n+1)
	ax.set_ylim([0, None])
	ax.legend(fontsize=legendsize)
	fig.tight_layout()

	print('Position w/ maximal error (1-idxed): %d' % (np.argmax(np.mean(errs_insert, axis=1))+1))

	plt.show()


if __name__ == '__main__':
	np.random.seed(0)
	simulate_err()
	simulate_err_per_item()
