# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:24:28 2021

@author: KEPetousakis
"""

import Code_General_utility_spikes_pickling as util
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dcp
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.colors as mcolors

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

mpl.rc('font', **font)

__RES_DPI = 48
_BONFERRONI = False

CNDS = [600]
testcnd = 600
BNFS = {x:[20] for x in CNDS}
NRNS = [x for x in range(0,10)]
RUNS = [x for x in range(0,10)]
BNFS_PLOT = {x:[0] for x in CNDS}
_CONDITIONS = {100:"10b:90a",200:"20b:80a",300:"30b:70a",400:"40b:60a",500:"50b:50a", 600:"60b:40a",700:"70b:30a",800:"80b:20a",900:"90b:10a"}

outcome_dict = {0:'apical',1:'basal',2:'unstable',3:'bistable'}

subplot_indices = {0:[0,0], 1:[1,0], 2:[0,1], 3:[1,1]}

def test_stats_add_stars(labels,data,feature,starheight,staroffset,offsetmove, comparewith=-1):
	# Statistical testing with paired two-sample t-test
	N = len(data)
	
	xx = [x for x in range(0, len(labels))]
	xraw = [x for x in range(0, len(labels))]
	
	results = {x:{y:0 for y in labels} for x in labels}  # table-style dict for p-values - comparisons are in the style of X to Y, accessing the result via results[x][y]
	res_array = np.zeros(shape = (N,N))
	res_array_bin = np.zeros(shape = (N,N))
	test = stats.ttest_ind
	verdict = lambda pval: '*' if pval <= 0.05 else 'NS'
	if comparewith == -1:
		comparator_labels = labels
	else:
		if type([]) != type(comparewith):
			comparewith = [comparewith]
		comparator_labels = [labels[x] for x in comparewith]
	for i,comparator in enumerate(comparator_labels):
		for j,comparand in enumerate(labels):
			if comparator == comparand:
				res_array[i][j] = -1
				res_array_bin[i][j] = -1
			else:
				(stat, pval) = test(data[i], data[j])
				print(f'Testing {comparator} against {comparand}. Results: Statistic {stat:2.4f}, p-value {pval:2.8f} | Verdict: \t {verdict(pval)}')
				results[comparator][comparand] = pval
				res_array[i][j] = pval
				if not _BONFERRONI:
					if pval <= 0.05:
						res_array_bin[i][j] = 1
					if pval <= 0.01:
						res_array_bin[i][j] = 2
					if pval <= 0.001:
						res_array_bin[i][j] = 3
					if pval > 0.05:
						res_array_bin[i][j] = 0
				else:
					alpha_bf = 0.05/(8)  # control compared with the rest is 8 comparisons, regardless of what is displayed in each subplot
					if pval <= alpha_bf:
						res_array_bin[i][j] = 1
					else:
						res_array_bin[i][j] = 0
				print(f'Testing {comparator} against {comparand}. Results: Statistic {stat:2.4f}, p-value {pval:2.4f} | Verdict: \t {"*" if len(int(res_array_bin[i][j])*"*")>0 else "NS"}')
				
	idxtolabel = {x:y for x,y in zip(xraw, labels)}
	idxtopos = {x:y for x,y in zip(xraw, xx)}
	
	results_triu = np.triu(res_array_bin)
	lineardata = []
	for datum in [x for x in data]:
		lineardata += [x for x in datum]
	starting_height = 1
	offset = 0.05
	
	for i in range(0,N):
		for j in range(0,N):
			if results_triu[i,j] > 0:
				offset += offsetmove
				print(f'>0 for i={i} and j={j} ( comparator={labels[i]} and comparand={labels[j]} )')
				height_diff = offset
				xvals = [idxtopos[i], idxtopos[j]]
				yvals = [starting_height-height_diff, starting_height-height_diff]
				plt.plot(xvals, yvals, 'k')
				print(f"Placing star at {np.mean(xvals)} and {yvals[0]}+")
				plt.text(np.mean(xvals), yvals[0], '*'*int(res_array_bin[i][j]), fontsize='xx-large', ha='center')  # ha = horizontal alignment
	return results


def lookup(target_dict, search_parameter):
	
	keys = target_dict.keys()
	matches = {}
	for key in keys:
		if search_parameter in key:
			matches[key] = target_dict[key]
	return matches

def verdictsbyorient(verdicts, orientation):
	
	keys = verdicts.keys()
	matches = {}
	for i,key in enumerate(keys):
		key_elements = key.split('-')
		if int(key_elements[3]) == orientation:
			matches[key] = verdicts[key]
	return matches

def autolabel(rects, axes, hoffset=0):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        axes.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(hoffset, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

_FIGURES = ['Figure 4B', 'Figure 4D', 'Figure 4C', 'Figure 4E']
filepath_root = './Data/F3/'

filepath_fixednoise = f'{filepath_root}/fixednoise/spike_survival_verdicts_revisions_na_intv.pickle'
filepath_variablenoise = f'{filepath_root}/variablenoise/spike_survival_verdicts_revisions_na_intv.pickle'

angle = -1

verdicts_vn = util.pickle_load(filepath_variablenoise)
verdicts_fn = util.pickle_load(filepath_fixednoise)

verdicts_vn_s0 = verdictsbyorient(verdicts_vn, 0)
verdicts_vn_s90= verdictsbyorient(verdicts_vn, 90)
verdicts_fn_s0 = verdictsbyorient(verdicts_fn, 0)
verdicts_fn_s90= verdictsbyorient(verdicts_fn, 90)

verdicts_all = [verdicts_vn_s0, verdicts_vn_s90, verdicts_fn_s0, verdicts_fn_s90]

per_nrn_spikes = np.zeros(shape = (len(verdicts_all), len(NRNS),  4))
cross_nrn_avg_spikes = np.zeros(shape = (len(verdicts_all), 4))
cross_nrn_std_spikes = np.zeros(shape = (len(verdicts_all), 4))
cross_nrn_serr_spikes = np.zeros(shape = (len(verdicts_all), 4))
cross_nrn_med_spikes = np.zeros(shape = (len(verdicts_all), 4))

for idx_v, verdicts in enumerate(verdicts_all):
	
	results = np.zeros( shape = (len(CNDS), len(NRNS), len(RUNS), 5) )  # 5 = 0(apical), 1(basal), 2(unstable), 3(bistable) - each holds a count (number of occurrences) - 5 is the sum
	summation = np.zeros( shape = (len(CNDS) ) )
	fractions = np.zeros( shape = (len(CNDS), 4) )
	fractions_per_neuron = np.zeros( shape = (len(CNDS), len(NRNS), len(RUNS), 4) )
	cross_nrn_averages = np.zeros( shape = (len(CNDS), 4) )
	cross_nrn_stds = np.zeros( shape = (len(CNDS),  4) )
	total_fractions = np.zeros( shape = (len(CNDS), 4))
	
	total_spikes = 0
	total_spikes_list = []
	cnd_sums = []
	
	for ic, cnd in enumerate(CNDS):
		for ib, bnf in enumerate(BNFS):
			cnd_sum = 0
			for nrn in NRNS:
				for run in RUNS:
					if angle < 0:
						search_parameter = f'{cnd}-{nrn}-{run}-'
					else:
						search_parameter = f'{cnd}-{nrn}-{run}-{angle}-'
					matches = lookup(verdicts, search_parameter)
					matches = [x for x in matches.values()]
					matches = [x['Verdict'] for x in matches]
					apicals   = sum(['apical' in x for x in matches])
					basals    = sum(['basal' in x for x in matches])
					unstables = sum(['unstable' in x for x in matches])
					bistables = sum(['bistable' in x for x in matches])
					
					results[ic][nrn][run][0] = apicals
					results[ic][nrn][run][1] = basals
					results[ic][nrn][run][2] = unstables
					results[ic][nrn][run][3] = bistables
					
					per_nrn_spikes[idx_v][nrn][0] += apicals
					per_nrn_spikes[idx_v][nrn][1] += basals
					per_nrn_spikes[idx_v][nrn][2] += unstables
					per_nrn_spikes[idx_v][nrn][3] += bistables
					
					sum_res = apicals+basals+unstables+bistables
					results[ic][nrn][run][4] = sum_res
					partialsum = apicals + basals + unstables + bistables
					cnd_sum += partialsum
					total_spikes += partialsum

					fractions[ic][0] += apicals
					fractions[ic][1] += basals
					fractions[ic][2] += unstables
					fractions[ic][3] += bistables
					summation[ic]    += sum_res
					
					if sum_res != 0:
						fractions_per_neuron[ic][nrn][run][0] = apicals/sum_res
						fractions_per_neuron[ic][nrn][run][1] = basals/sum_res
						fractions_per_neuron[ic][nrn][run][2] = unstables/sum_res
						fractions_per_neuron[ic][nrn][run][3] = bistables/sum_res
					else:
						fractions_per_neuron[ic][nrn][run][0] = 0
						fractions_per_neuron[ic][nrn][run][1] = 0
						fractions_per_neuron[ic][nrn][run][2] = 0
						fractions_per_neuron[ic][nrn][run][3] = 0
						
				# Fractionalization for averaging
				this_nrn_spike_sum = per_nrn_spikes[idx_v][nrn][0] + per_nrn_spikes[idx_v][nrn][1] + per_nrn_spikes[idx_v][nrn][2] + per_nrn_spikes[idx_v][nrn][3]
				if this_nrn_spike_sum > 0:
					per_nrn_spikes[idx_v][nrn][0] /= this_nrn_spike_sum
					per_nrn_spikes[idx_v][nrn][1] /= this_nrn_spike_sum
					per_nrn_spikes[idx_v][nrn][2] /= this_nrn_spike_sum
					per_nrn_spikes[idx_v][nrn][3] /= this_nrn_spike_sum
				else:
					per_nrn_spikes[idx_v][nrn][0] = 0
					per_nrn_spikes[idx_v][nrn][1] = 0
					per_nrn_spikes[idx_v][nrn][2] = 0
					per_nrn_spikes[idx_v][nrn][3] = 0
			
			cnd_sums.append(cnd_sum)
			
			if summation[ic] != 0:
				fractions[ic][0] /= summation[ic]				
				fractions[ic][1] /= summation[ic]
				fractions[ic][2] /= summation[ic]				
				fractions[ic][3] /= summation[ic]
			else:
				fractions[ic][0] = 0
				fractions[ic][1] = 0
				fractions[ic][2] = 0
				fractions[ic][3] = 0
			
				
		total_fractions[ic][0] = sum(sum(results[ic,:,:,0]))/sum(sum(results[ic,:,:,4]))
		total_fractions[ic][1] = sum(sum(results[ic,:,:,1]))/sum(sum(results[ic,:,:,4]))
		total_fractions[ic][2] = sum(sum(results[ic,:,:,2]))/sum(sum(results[ic,:,:,4]))
		total_fractions[ic][3] = sum(sum(results[ic,:,:,3]))/sum(sum(results[ic,:,:,4]))
		print(f'Total spikes {total_spikes}')
		total_spikes_list.append(total_spikes)

	
	# Added for across neuron averaging
	cross_nrn_avg_spikes[idx_v,:] = np.mean(per_nrn_spikes[idx_v], axis=0)
	cross_nrn_std_spikes[idx_v,:] = np.std(per_nrn_spikes[idx_v], axis=0)
	cross_nrn_med_spikes[idx_v,:] = np.median(per_nrn_spikes[idx_v], axis=0)
	cross_nrn_serr_spikes[idx_v,:] = np.std(per_nrn_spikes[idx_v], axis=0)/np.sqrt(10)

	print(f'Totals, {_FIGURES[idx_v]}:')
	print(f'Apical {cross_nrn_avg_spikes[idx_v,0]:.4f}+/-{cross_nrn_std_spikes[idx_v,0]:.4f}')
	print(f'Basal {cross_nrn_avg_spikes[idx_v,1]:.4f}+/-{cross_nrn_std_spikes[idx_v,1]:.4f}')
	print(f'Coop {cross_nrn_avg_spikes[idx_v,2]:.4f}+/-{cross_nrn_std_spikes[idx_v,2]:.4f}')
	print(f'Bistable {cross_nrn_avg_spikes[idx_v,3]:.4f}+/-{cross_nrn_std_spikes[idx_v,3]:.4f}')

	
	# Average across neurons to get mean + std
	for ic,cnd in enumerate(CNDS):
		for ib,bnf in enumerate(BNFS[cnd]):
			for outcome in range(0,4):
				cross_nrn_averages[ic][outcome] = np.mean(np.mean(fractions_per_neuron[ic,:,:,outcome], axis=1))
				cross_nrn_stds[ic][outcome] = np.std(np.mean(fractions_per_neuron[ic,:,:,outcome], axis=1)/np.sqrt(10))


	
	# Figures 4B, 4C, 4D, 4E
	labels = ['Apical','Basal','Cooperative']
	for ic, cnd in enumerate(CNDS):
		
		if idx_v == 0:
			fig, axs = plt.subplots(nrows=2, ncols=2, dpi=__RES_DPI)
			axs[0,0].set_ylabel('Stimulus 0$^\circ$'+'\n\nFraction of spikes')
			axs[1,0].set_ylabel('Stimulus 90$^\circ$'+'\n\nFraction of spikes')
			axs[0,0].set_title('Control background')
			axs[0,1].set_title('Evenly distributed background')
			
			axs[0,0].set_aspect('equal', 'box')
			axs[0,1].set_aspect('equal', 'box')
			axs[1,0].set_aspect('equal', 'box')
			axs[1,1].set_aspect('equal', 'box')
			
			
		this_res = total_fractions[ic,0:-1]
		avgd_fractions = cross_nrn_avg_spikes[idx_v][0:-1]
		fraction_serrs = cross_nrn_serr_spikes[idx_v][0:-1]
		

	
		x = np.arange(len(labels))
		x = [0.3, 0.6, 0.9]
		width = 0.20

		rects_this = axs[subplot_indices[idx_v][0],subplot_indices[idx_v][1]].bar(x, avgd_fractions, width, yerr=fraction_serrs, label = f'{_CONDITIONS[cnd]}', alpha = 0.7, capsize=5)
		axs[subplot_indices[idx_v][0],subplot_indices[idx_v][1]].set_xticks(x)
		axs[subplot_indices[idx_v][0],subplot_indices[idx_v][1]].set_xticklabels(labels)
		axs[subplot_indices[idx_v][0],subplot_indices[idx_v][1]].set_ylim([0,1])
		axs[subplot_indices[idx_v][0],subplot_indices[idx_v][1]].text(0.6, 0.9, f'N={total_spikes_list[ic]:.0f}',ha='center', fontsize=18)
		
		
		autolabel(rects_this,axs[subplot_indices[idx_v][0],subplot_indices[idx_v][1]],25)

	fig.set_tight_layout(True)
	figmanager = plt.get_current_fig_manager()
	figmanager.window.showMaximized()
	
plt.show()

	
	
# ======  Statistical Tests =======	

labels_all = ['vn_s0', 'vn_s90', 'fn_s0', 'fn_s90']  # vn: variable noise / fn: fixed (evenly distributed) noise / s0: stimulus@0deg / s90: stimulus@90deg
test_labels = [('vn_s0', 'vn_s90'), ('vn_s0', 'fn_s0'), ('vn_s90','fn_s90')]
test_indices = [(0,1),(0,2),(1,3)]
target_labels = ['apical', 'basal', 'cooperative', 'bistable']

test_res = { x:{ y:{ z:0 for z in target_labels } for y in labels_all } for x in labels_all }

test = stats.ttest_ind
	
for i,li in enumerate(labels_all):
	for j, lj in enumerate(labels_all):
		for k, lk in enumerate(target_labels):

			(stat, pval) = test(per_nrn_spikes[i,:,k], per_nrn_spikes[j,:,k])
			test_res[li][lj][lk] = (stat, pval)
			
print("Results:")
for i in range(0,3):
	print('')
	for j in range(0,4):
		s = test_res[test_labels[i][0]][test_labels[i][1]][target_labels[j]][0]
		pv = test_res[test_labels[i][0]][test_labels[i][1]][target_labels[j]][1]
		if pv < 0.05:
			if pv < 0.05:
				d = 1
			if pv <= 0.01:
				d = 2
			if pv <= 0.001:
				d = 3
			decision = '*'*d
		else:
			decision = 'NS'
		print(f'{test_labels[i][0]} vs {test_labels[i][1]} for {target_labels[j]}: stat {s:.4g}, pval {pv:.4f} : {decision}')

		

	
	