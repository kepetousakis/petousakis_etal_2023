# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:24:28 2021

@author: KEPetousakis
"""

import Code_General_utility_spikes_pickling as util
import numpy as np
import Code_General_Nassi_functions as nf
import matplotlib.pyplot as plt
from copy import deepcopy as dcp
import scipy.stats as stats
import matplotlib

import warnings
import logging

logging.getLogger('matplotlib.font_manager').disabled = True

warnings.filterwarnings("ignore")

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

__RES_DPI = 48

_eq_var = True
_handle_nan = 'omit' # 'propagate'

_BONFERRONI = False

def test_stats_add_stars_generic(labels,data,feature,starheight,staroffset,offsetmove, comparewith=-1):
	# Statistical testing with paired two-sample t-test
	N = len(data)
	
	xx = [x for x in range(0, len(labels))]
	xraw = [x for x in range(0, len(labels))]
	
	results = {x:{y:0 for y in labels} for x in labels}  # table-style dict for p-values - comparisons are in the style of X to Y, accessing the result via results[x][y]
	res_array = np.zeros(shape = (N,N))
	res_array_bin = np.zeros(shape = (N,N))
	test = stats.ttest_rel 
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
				(stat, pval) = test(data[i][feature], data[j][feature], nan_policy=_handle_nan) #,equal_var=_eq_var)
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
					alpha_bf = 0.05/(8)  # control compared with the rest is 8 comparisons
					if pval <= alpha_bf:
						res_array_bin[i][j] = 1
					else:
						res_array_bin[i][j] = 0
				# print(f'Testing {comparator} against {comparand}. Results: Statistic {stat:2.4f}, p-value {pval:2.4f} | Verdict: \t {"*" if len(int(res_array_bin[i][j])*"*")>0 else "NS"}')
				
	idxtolabel = {x:y for x,y in zip(xraw, labels)}
	idxtopos = {x:y for x,y in zip(xraw, xx)}
	
	results_triu = np.triu(res_array_bin)
	lineardata = []
	for datum in [x[feature] for x in data]:
		lineardata += [x for x in datum]
	starting_height = np.nanmax(lineardata) + starheight
	offset = staroffset
	
	for i in range(0,N):
		for j in range(0,N):
			if results_triu[i,j] > 0:
				offset += offsetmove
				# print(f'>0 for i={i} and j={j} ( comparator={labels[i]} and comparand={labels[j]} )')
				height_diff = starting_height -np.nanmax([data[i][feature],data[j][feature]]) -(offset)
				xvals = [idxtopos[i], idxtopos[j]]
				yvals = [starting_height-height_diff, starting_height-height_diff]
				plt.plot(xvals, yvals, 'k')
				# print(f"Placing star at {np.mean(xvals)} and {yvals[0]}+")
				plt.text(np.mean(xvals), yvals[0], '*'*int(res_array_bin[i][j]), fontsize='xx-large', ha='center')  # ha = horizontal alignment
	return results

def test_stats_add_stars(labels,data,feature,starheight,staroffset,offsetmove, comparewith=-1):
	# Statistical testing with paired two-sample t-test
	N = len(data)
	
	xx = [x for x in range(0, len(labels))]
	xraw = [x for x in range(0, len(labels))]
	
	results = {x:{y:0 for y in labels} for x in labels}  # table-style dict for p-values - comparisons are in the style of X to Y, accessing the result via results[x][y]
	res_array = np.zeros(shape = (N,N))
	res_array_bin = np.zeros(shape = (N,N))
	test = stats.ttest_rel #stats.ttest_ind
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
# 				(stat, pval) = test(data[i][feature], data[j][feature])
				(stat, pval) = test(data[i], data[j], nan_policy=_handle_nan) #,equal_var=_eq_var)
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
# 	starting_height = max(lineardata) + starheight
	offset = 0.05
	
	for i in range(0,N):
		for j in range(0,N):
			if results_triu[i,j] > 0:
				offset += offsetmove
				height_diff = offset
				xvals = [idxtopos[i], idxtopos[j]]
				yvals = [starting_height-height_diff, starting_height-height_diff]
				plt.plot(xvals, yvals, 'k')
				# print(f"Placing star at {np.mean(xvals)} and {yvals[0]}+")
				plt.text(np.mean(xvals), yvals[0], '*'*int(res_array_bin[i][j]), fontsize='xx-large', ha='center')  # ha = horizontal alignment
	return results

def lookup(target_dict, search_parameter):
	
	keys = target_dict.keys()
	matches = {}
	for key in keys:
		if search_parameter in key:
			matches[key] = target_dict[key]
	return matches

def eval_tuning(firing_rates_all_neurons_across_runs):
	adjusted_rates = firing_rates_all_neurons_across_runs
	prefs = []
	OSIs = []
	widths = []
	verdicts= []
	neurons = [x for x in range(0,np.min(np.shape(firing_rates_all_neurons_across_runs)))]
	global_rejections = 0
	for idx_n, nrn in enumerate(neurons):
		print(f'Neuron {nrn}', end = '  |  ')
		relevant_rates = np.squeeze(firing_rates_all_neurons_across_runs[idx_n,:])
		(nrn_pref, nrn_OSI, nrn_width, _ , _ ) = nf.tuning_properties(relevant_rates, [x*10 for x in range(0,18)])
		prefs.append(nrn_pref); OSIs.append(nrn_OSI); widths.append(nrn_width)
		if nrn_OSI < 0.2 or nrn_width > 80 or np.isnan(nrn_OSI):
			verdict = 'REJECT'
			global_rejections += 1
			adjusted_rates[idx_n,:] = np.nan
		else:
			verdict = 'ACCEPT'
		verdicts.append(verdict)
		print(f'Pref {nrn_pref} OSI {nrn_OSI:.3f} width {nrn_width} \t{verdict}')
		if verdict == 'REJECT':
			print('\t\t\t', end='')
			for x in relevant_rates:
				print(f'{x:.04}', end=', ')
			print()
	if global_rejections > 2:
		print(f'<!> Rejecting all neurons ({global_rejections}/10 neurons rejected).')
	else:
		print('<!> Neuron shows normal orientation tuning overall.')
		
	return adjusted_rates, prefs, OSIs, widths, verdicts

dt = 0.1
runtime = 2500
onset = 500

CND = 600
NRNS = [x for x in range(0,10)]
RUNS = [x for x in range(0,10)]
STIMS = [x*10 for x in range(0,18)]
CASES = ['Anull','Bnull']
_labels = ['Control', 'Apical target', 'Basal target']

_COLORS = ['r','g']

_CONDITIONS = {100:"10b:90a",200:"20b:80a",300:"30b:70a",400:"40b:60a",500:"50b:50a", 600:"60b:40a",700:"70b:30a",800:"80b:20a",900:"90b:10a"}

outcome_dict = {0:'apical',1:'basal',2:'unstable',3:'bistable'}

_TRANSFORM_VECTOR = np.array([9,8,7,6,5,4,3,2,1,0,17,16,15,14,13,12,11,10,9])
_TRANSFORM_VECTOR = np.flip(_TRANSFORM_VECTOR)

itype = 1 # 0 for sodium, 1 for synaptic

# Control data (sttx0, regular tuning curve)
ctrl_rates, ctrl_stds, _, _, ctrl_rates_all = util.pickle_load('./Data/F4/firing_rates_stds_cnd600.pickle')

if itype == 0 :
	raise Exception
if itype == 1:
	filepath = './Data/F4/intv_control_cnd600_delFB_nsA.pickle'
	directory = './Data/F4/'
	filepath2 = './Data/F4/intv_control_cnd600_delFB_nsB.pickle'
	affix = '_syn_removal'

nsA_rates, nsA_stds, _, _, nsA_rates_all = util.pickle_load(filepath)
nsB_rates, nsB_stds, _, _, nsB_rates_all = util.pickle_load(filepath2)

		
firing_rates = np.zeros(shape=(2,10,10,18)) # nullification(0=Anull,1=Bnull), neuron, run, stim

OSIs_all = np.zeros(shape=(2,10))
widths_all = np.zeros(shape=(2,10))

fig, axes = plt.subplot_mosaic('AAB', dpi = __RES_DPI)
axTC = axes['A']
axOSI = axes['B']
fig.set_tight_layout(True)

# Get OSIs for control runs
control_OSIs = []
control_widths = []
for nrn in range(0,10):
	(_, nrn_OSI, nrn_width, _ , _ ) = nf.tuning_properties(ctrl_rates[nrn], [x*10 for x in range(0,18)])
	control_OSIs.append(nrn_OSI)
	control_widths.append(nrn_width)
	
ctrl_OSI_mean = np.mean(control_OSIs)
ctrl_OSI_stderr = np.std(control_OSIs)/np.sqrt(10)

# Get OSIs for nsA runs
nsA_OSIs = []
nSA_widths = []
for nrn in range(0,10):
	(_, nsA_nrn_OSI, nsA_nrn_width, _ , _ ) = nf.tuning_properties(nsA_rates[nrn], [x*10 for x in range(0,18)])
	if nsA_nrn_OSI > 0.2 and nsA_nrn_width < 80:
		OSIs_all[0,nrn] = nsA_nrn_OSI
		widths_all[0,nrn] = nsA_nrn_width
	else:
		OSIs_all[0,nrn] = np.nan
		widths_all[0,nrn] = np.nan

nsA_rates_mean = np.mean(nsA_rates, axis=0)	
nsA_rates_serr = np.std(nsA_rates, axis=0)/np.sqrt(10)
nsA_OSI_mean = np.nanmean(nsA_OSIs)
nsA_OSI_stderr = np.nanstd(nsA_OSIs)/np.sqrt(10)

# Get OSIs for control runs
nsB_OSIs = []
nSB_widths = []
for nrn in range(0,10):
	(_, nsB_nrn_OSI, nsB_nrn_width, _ , _ ) = nf.tuning_properties(nsB_rates[nrn], [x*10 for x in range(0,18)])
	if nsB_nrn_OSI > 0.2 and nsB_nrn_width < 80:
		OSIs_all[1,nrn] = nsB_nrn_OSI
		widths_all[1,nrn] = nsB_nrn_width
	else:
		OSIs_all[1,nrn] = np.nan
		widths_all[1,nrn] = np.nan
	
nsB_rates_mean = np.mean(nsB_rates, axis=0)	
nsB_rates_serr = np.std(nsB_rates, axis=0)/np.sqrt(10)
nsB_OSI_mean = np.nanmean(nsB_OSIs)
nsB_OSI_stderr = np.nanstd(nsB_OSIs)/np.sqrt(10)

exp_rates = [nsA_rates_mean, nsB_rates_mean]
exp_serr = [nsA_rates_serr, nsB_rates_serr]

for iC, case in enumerate(CASES):
	print(f'Case {case}')

	# Remove untuned OSIs
	for exp in OSIs_all:
		if np.sum([1 if np.isnan(x) else 0 for x in exp]) > 2:
			OSIs_all[iC,:] = np.nan
			widths_all[iC,:] = np.nan
		else:
			pass
			
	
	x_axis = np.array([x*10 for x in range(-9,10)])
	y_axis = np.array(exp_rates[iC])
	y_errors = np.array(exp_serr[iC])
	
	preference = np.argmax(y_axis)*10
	
	print(f'Mean preferred orientation: {preference}')
	
	axTC.errorbar(x_axis, y_axis[_TRANSFORM_VECTOR], y_errors[_TRANSFORM_VECTOR], label=f'{_labels[iC+1]}', c=f'{_COLORS[iC]}', capsize=5)
	
	if iC == 0:
		axTC.errorbar(x_axis, np.mean(ctrl_rates, axis=0)[_TRANSFORM_VECTOR], np.std(ctrl_rates, axis=0)[_TRANSFORM_VECTOR]/np.sqrt(10), label=f'{_labels[0]}', capsize=5)
		axTC.set_xticks(x_axis)
		axTC.set_ylim([0,2])
		
		axOSI.bar(0, ctrl_OSI_mean, width=0.2, align='center', yerr=ctrl_OSI_stderr, capsize=5)
	
print(f'Color {_COLORS[iC]} (case {case})')
axOSI.bar([x for x in range(1,len(CASES)+1)], np.nanmean(OSIs_all, axis=1), width=0.2, align='center', yerr=np.nanstd(OSIs_all, axis=1)/np.sqrt(10), color=['r','g'], capsize=5)
axOSI.set_xticks([x for x in range(0,len(CASES)+1)],_labels)
axOSI.set_ylabel('Mean OSI value')
axOSI.set_ylim([0,1.05])
	
	
OSIs_control = control_OSIs[:]
OSIs_Anull = [x for x in OSIs_all[0,:]]
OSIs_Bnull = [x for x in OSIs_all[1,:]]
OSIs_aggregated = [[OSIs_control], [OSIs_Anull], [OSIs_Bnull]]
plt.sca(axOSI)
for comparison in range(0,1):
	# 0: control to Anull, 1: control to Bnull, 2: Anull to Bnull
	res = test_stats_add_stars_generic(_labels,OSIs_aggregated,0,0.07,0,0.05,comparewith=[0,1,2])
		
	
axTC.set_title(f'Figure 4A')
axTC.set_xlabel('Stimulus orientation ($^\circ$)')
axTC.set_ylabel('Neuronal response (Hz)')

axOSI.set_xlim([-0.2,2.2])
	
axTC.legend()
figmanager = plt.get_current_fig_manager()
figmanager.window.showMaximized()
plt.show()



