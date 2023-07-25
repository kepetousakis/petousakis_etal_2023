import pandas as pd
import Code_General_utility_spikes_pickling as util
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats


_RES_DPI = 48

_SAVEDATA = False

_X_MARKER_SIZE = 50
_show_excluded = False

ID_trunk = [0,1,2,3,4]  # IDs of trunk dendrites (add 7 to ID to return linear index)
ID_atypical = [21,18,29,36]  # IDs of atypical dendrites (add 7 to ID to return linear index)

dt=0.1
dendrite_information = {}

basal_names = ['basal'+f'{x}' for x in range(0,7)]
apical_names = ['apical'+f'{x}' for x in range(0,43)]
dendrite_names = basal_names + apical_names

basal_range = [x for x in range(0,7)]
apical_range = [x for x in range(7,50)]

filepath = './Data/Data_dendrite_properties.tsv'
info = pd.read_csv(filepath, sep='\t')

for i, segname in enumerate(dendrite_names):
	idx = i+2 # to skip soma and axon
	# "ELP" or "ELC" in this code stands for "Electrotonic (Length) Constant"
	dendrite_information[segname] = {'diameter':info['diam'][idx],'length':info['L'][idx],'distance':info['totlength'][idx],'ELP':1/info['elp'][idx],'tELP':info['telp'][idx],'order':info['nc'][idx],'terminal':False,'name':segname,'idx':i}


# Input of dendrite information that we can't load from the dataframe
# Basal dendrites
dendrite_information['basal0']['terminal'] = True
dendrite_information['basal1']['terminal'] = True
dendrite_information['basal2']['terminal'] = True
dendrite_information['basal3']['terminal'] = False
dendrite_information['basal4']['terminal'] = True
dendrite_information['basal5']['terminal'] = True
dendrite_information['basal6']['terminal'] = True

# Apical trunk dendrites
dendrite_information['apical0']['terminal'] = False
dendrite_information['apical1']['terminal'] = False
dendrite_information['apical2']['terminal'] = False

# Left side apical tuft dendrites
dendrite_information['apical3']['terminal'] = False
dendrite_information['apical4']['terminal'] = False
dendrite_information['apical5']['terminal'] = False
dendrite_information['apical6']['terminal'] = False
dendrite_information['apical7']['terminal'] = True
dendrite_information['apical8']['terminal'] = True
dendrite_information['apical9']['terminal'] = True
dendrite_information['apical10']['terminal'] = False
dendrite_information['apical11']['terminal'] = False
dendrite_information['apical12']['terminal'] = True
dendrite_information['apical13']['terminal'] = True
dendrite_information['apical14']['terminal'] = False
dendrite_information['apical15']['terminal'] = True
dendrite_information['apical16']['terminal'] = True
dendrite_information['apical17']['terminal'] = True
dendrite_information['apical18']['terminal'] = False
dendrite_information['apical19']['terminal'] = True
dendrite_information['apical20']['terminal'] = True

# Right side apical tuft dendrites
dendrite_information['apical21']['terminal'] = False
dendrite_information['apical22']['terminal'] = False
dendrite_information['apical23']['terminal'] = False
dendrite_information['apical24']['terminal'] = True
dendrite_information['apical25']['terminal'] = False
dendrite_information['apical26']['terminal'] = True
dendrite_information['apical27']['terminal'] = True
dendrite_information['apical28']['terminal'] = True
dendrite_information['apical29']['terminal'] = False
dendrite_information['apical30']['terminal'] = False
dendrite_information['apical31']['terminal'] = False
dendrite_information['apical32']['terminal'] = True
dendrite_information['apical33']['terminal'] = True
dendrite_information['apical34']['terminal'] = True
dendrite_information['apical35']['terminal'] = True

# Oblique apical dendrites
dendrite_information['apical36']['terminal'] = False
dendrite_information['apical37']['terminal'] = False
dendrite_information['apical38']['terminal'] = True
dendrite_information['apical39']['terminal'] = True
dendrite_information['apical40']['terminal'] = False
dendrite_information['apical41']['terminal'] = True
dendrite_information['apical42']['terminal'] = True

# Extract branch order only for all segments
branch_orders = []
for i,segname in enumerate(dendrite_names):
	branch_orders.append(dendrite_information[segname]['order'])
	
ELP_values = []
for i,segname in enumerate(dendrite_names):
	ELP_values.append(dendrite_information[segname]['ELP'])
		
	
tELP_values = []
for i,segname in enumerate(dendrite_names):
	tELP_values.append(dendrite_information[segname]['tELP'])
	

diameter_values = []
for i,segname in enumerate(dendrite_names):
	diameter_values.append(dendrite_information[segname]['diameter'])
	

# Neuron connectivity matrix
a2i = lambda x: x+7
b2i = lambda x: x
conn_matrix = np.zeros(shape=(51,51))

def A(x,y):
	global conn_matrix
	conn_matrix[x][y] = 1
	return

# Compartment indexing
a0 = a2i(0); a1 = a2i(1); a2 = a2i(2); a3 = a2i(3); a4 = a2i(4);
a5 = a2i(5); a6 = a2i(6); a7 = a2i(7); a8 = a2i(8); a9 = a2i(9);
a10 = a2i(10); a11 = a2i(11); a12 = a2i(12); a13 = a2i(13);
a14 = a2i(14); a15 = a2i(15); a16 = a2i(16); a17 = a2i(17);
a18 = a2i(18); a19 = a2i(19); a20 = a2i(20); a21 = a2i(21);
a22 = a2i(22); a23 = a2i(23); a24 = a2i(24); a25 = a2i(25);
a26 = a2i(26); a27 = a2i(27); a28 = a2i(28); a29 = a2i(29); 
a30 = a2i(30); a31 = a2i(31); a32 = a2i(32); a33 = a2i(33);
a34 = a2i(34); a35 = a2i(35); a36 = a2i(36); a37 = a2i(37);
a38 = a2i(38); a39 = a2i(39); a40 = a2i(40); a41 = a2i(41);
a42 = a2i(42);
b0 = b2i(0); b1 = b2i(1); b2 = b2i(2); b3 = b2i(3);
b4 = b2i(4); b5 = b2i(5); b6 = b2i(6);
soma = 50;

# Somatic connectivity
A(soma,b2)
A(soma,b6)
A(soma,b1)
A(soma,b0)
A(soma,b3)
A(soma,a0)

# Basal connectivity
A(b3,b4)
A(b3,b5)

A(b4,b5)

# Apical connectivity
# Trunk to proximal oblique
A(a0,a1)
A(a0,a36)

A(a36,a37)
A(a36,a40)

A(a37,a38)
A(a37,a39)
A(a37,a40)

A(a38,a39)

A(a40,a41)
A(a40,a42)

A(a41,a42)

# Trunk to rightmost apical tuft
A(a1,a2)
A(a1,a21)
A(a1,a36)

A(a21,a22)
A(a21,a29)

A(a22,a28)
A(a22,a23)
A(a22,a29)

A(a23,a24)
A(a23,a25)
A(a23,a28)

A(a24,a25)

A(a25,a26)
A(a25,a27)

A(a26,a27)

A(a29,a30)
A(a29,a35)

A(a30,a35)

A(a30,a31)
A(a30,a34)

A(a31,a34)
A(a31,a32)
A(a31,a33)

A(a32,a33)

# Trunk to middle apical tuft (3 branches)
A(a2,a3)
A(a2,a18)
A(a2,a21)
A(a2,a19)

A(a18,a19)
A(a18,a20)

A(a19,a20)

# Trunk to lefttmost apical tuft
A(a3,a18)
A(a3,a4)
A(a3,a17)

A(a4,a17)
A(a4,a5)
A(a4,a10)

A(a5,a10)
A(a5,a6)
A(a5,a9)

A(a6,a9)
A(a6,a7)
A(a6,a8)

A(a7,a8)

A(a10,a11)
A(a10,a14)

A(a11,a14)
A(a11,a12)
A(a11,a13)

A(a12,a13)

A(a14,a15)
A(a14,a16)

A(a15,a16)

# Enforce diagonal symmetry of conn. matrix (A -> B means B -> A as well)
for i in range(0,51):
	for j in range(0,51):
		if conn_matrix[i][j] == 1 or conn_matrix[j][i] == 1:
			A(i,j)
			A(j,i)


def find_paths(conn_matrix, dendrite_info):
	'''Starting from terminal dendrites, find the paths to the soma, and return them'''
	
	terminal_segments = [dendrite_info[key]['idx'] for key in dendrite_info.keys() if dendrite_info[key]['terminal']]
	info_by_index = [dendrite_information[key] for key in dendrite_information.keys()]
	
	paths = []
	
	for terminus in terminal_segments:
		tgt = terminus
		reached_soma = False
		path = []
		while not reached_soma:
			if conn_matrix[int(tgt)][50] != 1:
				neighbors = conn_matrix[tgt]; neighbors = [pos for pos,x in enumerate(neighbors) if x==1]
				n_dists = [info_by_index[int(idx)]['distance'] for idx in neighbors]
				min_dist_idx = np.argmin(n_dists)
				target_idx = neighbors[min_dist_idx]
				path.append(int(tgt))
				tgt = target_idx
			else:
				path.append(int(tgt))
				path.append(50)
				paths.append(path)
				reached_soma = True
				break
			
	return paths

def path_verbose(path, dendrite_names):
	'''Translate paths to the soma from indices to compartment names'''
	
	dendrite_names.append('soma')
	verbose_path = []
	for element in path:
		verbose_path.append(dendrite_names[element])
	return verbose_path

def inspect_path(term_dend_of_path, na_threshold):
	'''Returns a wave plot of voltages, starting from the compartment "term_dend_of_path", all the way to the soma.'''
	fig = plt.figure(dpi=_RES_DPI)
	tgt = term_dend_of_path
	info_by_index = [dendrite_information[key] for key in dendrite_information.keys()]
	p = []
	reached_soma = False
	compartment_names = dendrite_names + ['soma']
	
	show_basal_branches = False
	if compartment_names[tgt] == 'basal3':
		print('Also showing basal3 child branches.')
		show_basal_branches = True
	
	# reconstruct path to soma
	while not reached_soma:
		if conn_matrix[int(tgt)][50] != 1:
			neighbors = conn_matrix[tgt]; neighbors = [pos for pos,x in enumerate(neighbors) if x==1]
			n_dists = [info_by_index[int(idx)]['distance'] for idx in neighbors]
			min_dist_idx = np.argmin(n_dists)
			target_idx = neighbors[min_dist_idx]
			p.append(int(tgt))
			tgt = target_idx
		else:
			p.append(int(tgt))
			p.append(50)
			reached_soma = True
			break
	
	# image the path voltages
	
	if show_basal_branches:
		if 4 not in p:
			p.append(4)
		if 5 not in p:
			p.append(5)
	
	data = []
	amps = []
	print(f'From origin {term_dend_of_path}, found path {p} to soma (50){", including basal branches" if show_basal_branches else ""}.')

	for c in p:
		data.append(np.loadtxt(f'./data/attenuation/dendrite_{p[0]}_nstim_{na_threshold}/{compartment_names[c]}.dat'))
		# print(np.max(data[-1][0:int(220/dt)]))
		# print(data[-1][198])
		amps.append(np.max(data[-1][0:int(220/dt)])-data[-1][198])
		if compartment_names[c] == 'soma':
			soma_max_t = np.argmax(data[-1][1900:2401])
			
	for i,datum in enumerate(data):
		plt.plot(datum[1900:2401]+(i*50))
		plt.text(0,np.mean(datum+(i*50)),f'{compartment_names[p[i]]}: amp={amps[i]:.04f} mV')
	plt.axvline(np.argmax(data[0][1900:2401]),color='m', ls='--')
	plt.axvline(soma_max_t,color='b', ls='--')
	plt.axvline(101,color='k', ls=':')
	plt.axvline(301,color='k', ls=':')
	
	fig.set_tight_layout(True)
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	
	return fig

def find_path(term_dend_of_path):
	'''Returns a unique path to the soma, starting from the compartment "term_dend_of_path"'''
	tgt = term_dend_of_path
	info_by_index = [dendrite_information[key] for key in dendrite_information.keys()]
	p = []
	reached_soma = False
	compartment_names = dendrite_names + ['soma']
	
	# reconstruct path to soma
	while not reached_soma:
		if conn_matrix[int(tgt)][50] != 1:
			neighbors = conn_matrix[tgt]; neighbors = [pos for pos,x in enumerate(neighbors) if x==1]
			n_dists = [info_by_index[int(idx)]['distance'] for idx in neighbors]
			min_dist_idx = np.argmin(n_dists)
			target_idx = neighbors[min_dist_idx]
			p.append(int(tgt))
			tgt = target_idx
		else:
			p.append(int(tgt))
			p.append(50)
			reached_soma = True
			break

	return p

def visualize_compare_data(amp_data, data_apicals, data_basals, title_text='', range_std_mult=2, ID_trunk=ID_trunk, ID_atypical=ID_atypical, filter_trunk_atypical=True, filter_non_corresponding=True, assume_equal_variances=False):

	fig, axes = plt.subplot_mosaic('AAB', dpi=_RES_DPI)
	axScatter = axes['A']
	axBoxplot = axes['B']
	plt.sca(axScatter)
	plt.title(f'{title_text}')

	valid_apicals = np.array([int(x) for x in range(0,43)])
	invalid_apicals = np.array([])

	if filter_trunk_atypical:
		data_trunk = data_apicals[ID_trunk]
		data_atypical = data_apicals[ID_atypical]
		valid_apicals = np.array([i for i,x in enumerate(data_apicals) if (i not in ID_trunk and i not in ID_atypical)])
		if _show_excluded:
			plt.scatter(data_trunk, amp_data[ID_trunk,2], label='trunk',s=_X_MARKER_SIZE, marker='x', c='c')
			plt.scatter(data_atypical, amp_data[ID_atypical,2], label='atypical',s=_X_MARKER_SIZE, marker='x', c='g')
	if filter_non_corresponding:
		avg_b = np.mean(data_basals)
		std_b = np.std(data_basals)
		permissible_range = [avg_b-(range_std_mult*std_b), avg_b+(range_std_mult*std_b)]
		if _show_excluded:
			plt.axvline(permissible_range[0], ls='--', c='k', alpha=0.5)
			plt.axvline(permissible_range[1], ls='--', c='k', alpha=0.5)
		valid_apicals2 = np.array([i for i in valid_apicals if data_apicals[i]>=permissible_range[0] and data_apicals[i]<=permissible_range[1]])
		invalid_apicals = np.array([i for i in valid_apicals if data_apicals[i]<permissible_range[0] or data_apicals[i]>permissible_range[1]])
		valid_apicals = np.array(valid_apicals2)
	plt.scatter(data_basals, amp_data[0:7,2],c='b', label='basal')
	plt.scatter(data_apicals[valid_apicals], amp_data[7:,2][valid_apicals],c='r', label='apical')
	if len(invalid_apicals) > 0 and _show_excluded:
		plt.scatter(data_apicals[invalid_apicals], amp_data[7:,2][invalid_apicals],c='r', label='apical(excluded)', marker='x', s=_X_MARKER_SIZE)

	plt.legend()
	fig.set_tight_layout(True)
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	# figManager.set_window_title(title_text)
	
	plt.sca(axBoxplot)

	norm_basals = [x/y for x,y in zip(amp_data[0:7,2], data_basals)]
	norm_apicals = [x/y for x,y in zip(amp_data[7:,2][valid_apicals], data_apicals[valid_apicals])]
	
	stat, pval = test(np.array(norm_basals),np.array(norm_apicals), equal_var=assume_equal_variances)
	print(f'T-test statistic ({title_text}): {stat:.04f} | p-value: {pval:.06f} ({"not " if not assume_equal_variances else ""}assuming equal variances) | {"NS" if pval>=0.05 else "*"}')
	
	plt.title(f'T-test statistic: {stat:.04f} | p-value: {pval:.06f} | {"NS" if pval>=0.05 else "*"}\n({"not " if not assume_equal_variances else ""}assuming equal variances)')
	plt.boxplot([norm_basals, norm_apicals])
	plt.xticks([1,2],['basal', 'apical'])
	
	fig.set_tight_layout(True)
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	
	return fig


# ==== Active (spike) attenuation during a validation protocol for sodium spikes ====

if __name__ == '__main__':
	
	process_raw_data = False
	filter_trunk_atypical = True
	filter_non_corresponding = True
	assume_equal_variances = False
	range_std_mult = 3

	(na_thresholds,nmda_thresholds,na_maxima,na_maxima_backup,nmda_na_maxima,dendrite_NRLEs,traces_to_visualize_basal,traces_to_visualize_apical) = util.pickle_load('./Data/F1/Data_F1_F5_S1.pickle')
	
	# Need to compute attenuation along all apical and all basal paths 
	paths = find_paths(conn_matrix, dendrite_information)
	
	compartment_names = dendrite_names + ['soma']
			
	amp_data = util.pickle_load('./Data/F5/amp_data_thresholds.pickle')

	test = stats.ttest_ind	

	index_override = 0
	
	
# ===== Dendritic amplitudes (no other var) =====
# 	fig, axes = plt.subplot_mosaic('AAB', dpi=_RES_DPI)
# 	axScatter = axes['A']
# 	axBoxplot = axes['B']
# 	plt.sca(axScatter)
# 	plt.title('Dendritic amplitudes')
# 	range_basals = [x for x in range(0,7)]
# 	range_apicals = [x for x in range(7,50)]
# 	idx = 0
# 	if index_override:
# 		idx = index_override
# 	plt.scatter(range_basals, amp_data[0:7,idx],c='b', label='basal')
# 	plt.scatter(range_apicals, amp_data[7:,idx],c='r', label='apical')
# 	plt.legend()
# 	
# 	plt.sca(axBoxplot)
# 	stat, pval = test(np.array(amp_data[0:7,idx]),np.array(amp_data[7:,idx]), equal_var=False)
# 	print(f'T-test statistic (basal vs apical dendritic depolarization): {stat:.04f} | p-value: {pval:.06f} | {"NS" if pval>=0.05 else "*"}')
# 	plt.title(f'T-test statistic (basal vs apical dendritic depolarization):\n {stat:.04f} | p-value: {pval:.06f} | {"NS" if pval>=0.05 else "*"}')
# 	plt.boxplot([np.array(amp_data[0:7,0]),np.array(amp_data[7:,0])])
# 	
# 	fig.set_tight_layout(True)
# 	figManager = plt.get_current_fig_manager()
# 	figManager.window.showMaximized()
# 	
# 	
# 	# ===== Somatic amplitudes (no other var) =====
# 	fig, axes = plt.subplot_mosaic('AAB', dpi=_RES_DPI)
# 	axScatter = axes['A']
# 	axBoxplot = axes['B']
# 	plt.sca(axScatter)
# 	plt.title('Somatic amplitudes')
# 	idx = 1
# 	if index_override:
# 		idx = index_override
# 	plt.scatter(range_basals, amp_data[0:7,idx],c='b', label='basal')
# 	plt.scatter(range_apicals, amp_data[7:,idx],c='r', label='apical')
# 	plt.legend()
# 	
# 	plt.sca(axBoxplot)
# 	stat, pval = test(np.array(amp_data[0:7,idx]),np.array(amp_data[7:,idx]), equal_var=False)
# 	print(f'T-test statistic (basal vs apical somatic depolarization): {stat:.04f} | p-value: {pval:.06f} | {"NS" if pval>=0.05 else "*"}')
# 	plt.title(f'T-test statistic (basal vs apical somatic depolarization):\n {stat:.04f} | p-value: {pval:.06f} | {"NS" if pval>=0.05 else "*"}')
# 	plt.boxplot([np.array(amp_data[0:7,1]),np.array(amp_data[7:,1])])
# 	
# 	fig.set_tight_layout(True)
# 	figManager = plt.get_current_fig_manager()
# 	figManager.window.showMaximized()
# 	
# 	
# 	# ===== Threshold difference (no other var) =====
# 	fig, axes = plt.subplot_mosaic('AAB', dpi=_RES_DPI)
# 	axScatter = axes['A']
# 	axBoxplot = axes['B']
# 	plt.sca(axScatter)
# 	plt.title('Spiking Thresholds')
# 	idx = 2
# 	if index_override:
# 		idx = index_override
# 	plt.scatter(range_basals, amp_data[0:7,idx],c='b', label='basal')
# 	plt.scatter(range_apicals, amp_data[7:,idx],c='r', label='apical')
# 	plt.legend()
# 	
# 	plt.sca(axBoxplot)
# 	stat, pval = test(np.array(amp_data[0:7,idx]),np.array(amp_data[7:,idx]), equal_var=False)
# 	print(f'T-test statistic (basal vs apical spiking thresholds): {stat:.04f} | p-value: {pval:.06f} | {"NS" if pval>=0.05 else "*"}')
# 	plt.title(f'T-test statistic (basal vs apical spiking thresholds):\n {stat:.04f} | p-value: {pval:.06f} | {"NS" if pval>=0.05 else "*"}')
# 	plt.boxplot([np.array(amp_data[0:7,2]),np.array(amp_data[7:,2])])

# 	fig.set_tight_layout(True)
# 	figManager = plt.get_current_fig_manager()
# 	figManager.window.showMaximized()
	
	
	# ===== Distance-normalized attenuation between apical and basal dendrites =====
	range_basals = [x for x in range(0,7)]
	range_apicals = [x for x in range(7,50)]
	dist_basals = [dendrite_information[compartment_names[x]]['distance'] for x in range_basals]
	dist_apicals = [dendrite_information[compartment_names[x]]['distance'] for x in range_apicals]

	data_basals = np.array(dist_basals)
	data_apicals = np.array(dist_apicals)
	
	fig = visualize_compare_data(amp_data, data_apicals, data_basals, title_text='Threshold difference as a function of distance from the soma',range_std_mult=range_std_mult,filter_trunk_atypical=filter_trunk_atypical, filter_non_corresponding=filter_non_corresponding, assume_equal_variances=assume_equal_variances)

	

	# ===== Length-normalized attenuation between apical and basal dendrites =====
	len_basals = [dendrite_information[compartment_names[x]]['length'] for x in range_basals]
	len_apicals = [dendrite_information[compartment_names[x]]['length'] for x in range_apicals]
	
	data_basals = np.array(len_basals)
	data_apicals = np.array(len_apicals)
	
	fig = visualize_compare_data(amp_data, data_apicals, data_basals, title_text='Threshold difference as a function of dendrite length',range_std_mult=range_std_mult,filter_trunk_atypical=filter_trunk_atypical, filter_non_corresponding=filter_non_corresponding, assume_equal_variances=assume_equal_variances)


	# ===== Diameter-normalized attenuation between apical and basal dendrites =====
	diam_basals = [dendrite_information[compartment_names[x]]['diameter'] for x in range_basals]
	diam_apicals = [dendrite_information[compartment_names[x]]['diameter'] for x in range_apicals]
	
	data_basals = np.array(diam_basals)
	data_apicals = np.array(diam_apicals)

	fig = visualize_compare_data(amp_data, data_apicals, data_basals, title_text='Threshold difference as a function of dendritic diameter',range_std_mult=range_std_mult,filter_trunk_atypical=filter_trunk_atypical, filter_non_corresponding=filter_non_corresponding, assume_equal_variances=assume_equal_variances)


	# ===== ELC-normalized attenuation between apical and basal dendrites =====
	elc_basals = [dendrite_information[compartment_names[x]]['ELP'] for x in range_basals]
	elc_apicals = [dendrite_information[compartment_names[x]]['ELP'] for x in range_apicals]
	
	data_basals = np.array(elc_basals)
	data_apicals = np.array(elc_apicals)
	
	fig = visualize_compare_data(amp_data, data_apicals, data_basals, title_text='Figure 5C',range_std_mult=range_std_mult,filter_trunk_atypical=filter_trunk_atypical, filter_non_corresponding=filter_non_corresponding, assume_equal_variances=assume_equal_variances)
	

	# ===== Volume-normalized attenuation between apical and basal dendrites =====
	diam_basals = [dendrite_information[compartment_names[x]]['diameter'] for x in range_basals]
	diam_apicals = [dendrite_information[compartment_names[x]]['diameter'] for x in range_apicals]
	len_basals = [dendrite_information[compartment_names[x]]['length'] for x in range_basals]
	len_apicals = [dendrite_information[compartment_names[x]]['length'] for x in range_apicals]
	vol = lambda l,d: (np.pi*(d/2)**2)*l
	vol_basals = np.array([vol(x,y) for x,y in zip(len_basals,diam_basals)])
	vol_apicals = np.array([vol(x,y) for x,y in zip(len_apicals,diam_apicals)])
	
	data_basals = np.array(vol_basals)
	data_apicals = np.array(vol_apicals)
	
	fig = visualize_compare_data(amp_data, data_apicals, data_basals, title_text='Figure 5D',range_std_mult=range_std_mult,filter_trunk_atypical=filter_trunk_atypical, filter_non_corresponding=filter_non_corresponding, assume_equal_variances=assume_equal_variances)
	
	
	# ===== Path-volume-normalized attenuation between apical and basal dendrites =====
	diam_basals = [dendrite_information[compartment_names[x]]['diameter'] for x in range_basals]
	diam_apicals = [dendrite_information[compartment_names[x]]['diameter'] for x in range_apicals]
	len_basals = [dendrite_information[compartment_names[x]]['length'] for x in range_basals]
	len_apicals = [dendrite_information[compartment_names[x]]['length'] for x in range_apicals]
	vol = lambda l,d: (np.pi*(d/2)**2)*l
	vol_basals = np.array([vol(x,y) for x,y in zip(len_basals,diam_basals)])
	vol_apicals = np.array([vol(x,y) for x,y in zip(len_apicals,diam_apicals)])

	vol_all = np.concatenate((vol_basals, vol_apicals))

	path_volumes = np.zeros(shape=(50))
	for dend in range(0,50):
		p = find_path(dend)
		for branch in p:
			if p != 50:
				path_volumes[dend] += vol_all[dend]
			else:
				break
	
	path_vol_basals = path_volumes[0:7]
	path_vol_apicals = path_volumes[7:]
	
	data_basals = np.array(path_vol_basals)
	data_apicals = np.array(path_vol_apicals)
	
	fig = visualize_compare_data(amp_data, data_apicals, data_basals, title_text='Threshold difference as a function of path volume',range_std_mult=range_std_mult,filter_trunk_atypical=filter_trunk_atypical, filter_non_corresponding=filter_non_corresponding, assume_equal_variances=assume_equal_variances)
	
	

#=== Terminals-to-soma attenuation trajectories ===
	
# 	attenuation_trajectories = []
# 	
# 	for fullpath in paths:
# 		attenuation_traj = []
# 		tgt = fullpath[0]
# 		thr = na_thresholds[fullpath[0]]
# 		for comp in fullpath:
# 			trace = np.loadtxt(f'./data/attenuation/dendrite_{tgt}_nstim_{thr}/{compartment_names[comp]}.dat')
# 			baseline = trace[int(198/dt)]
# 			amp = np.max(trace[0:int(220/dt)]) - baseline
# 			attenuation_traj.append(amp)
# 		attenuation_trajectories.append(attenuation_traj)
# 	
# 	fig, axs = plt.subplots(nrows=2, ncols=3, dpi=48)
# 	fig.set_tight_layout(True)
# 	figManager = plt.get_current_fig_manager()
# 	figManager.window.showMaximized()	
# 	for i,ax in enumerate(axs[np.unravel_index(np.arange(6), axs.shape, 'F')]):
# 		plt.sca(ax)
# 		plt.plot(np.arange(len(paths[i])), attenuation_trajectories[i], '-D', zorder=1)
# 		plt.title(f'Basal path {paths[i]}')
# 		plt.xticks(np.arange(len(paths[i])), [compartment_names[x] for x in paths[i]], fontsize=12, rotation=15)
# 		plt.ylim([0,100])
# 		plt.axhline(attenuation_trajectories[i][-1],color='k', ls='--', zorder=0)
# 		plt.text(0, attenuation_trajectories[i][-1]+2.5, f'{attenuation_trajectories[i][-1]:.02f}', fontsize=14)
# 				
# 	
# 	fig, axs = plt.subplots(nrows=5, ncols=5, dpi=48)	
# 	fig.set_tight_layout(True)
# 	figManager = plt.get_current_fig_manager()
# 	figManager.window.showMaximized()
# 	for i, ax in enumerate(axs[np.unravel_index(np.arange(25), axs.shape, 'F')]):
# 		j = i+6
# 		if j < len(attenuation_trajectories):
# 			plt.sca(ax)
# 			plt.plot(np.arange(len(paths[j])), attenuation_trajectories[j], '-D', zorder=1)
# 			plt.title(f'Apical path {paths[j]}')
# 			plt.xticks(np.arange(len(paths[j])), [compartment_names[x] for x in paths[j]], fontsize=8, rotation=15)
# 			plt.ylim([0,100])
# 			plt.axhline(attenuation_trajectories[j][-1],color='k', ls='--', zorder=0)
# 			plt.text(0, attenuation_trajectories[j][-1]+2.5, f'{attenuation_trajectories[j][-1]:.02f}', fontsize=14)
# 			
# 	inspect_path(3,na_thresholds[3])		
# 	inspect_path(14,na_thresholds[14])
		
	plt.show()
	
	