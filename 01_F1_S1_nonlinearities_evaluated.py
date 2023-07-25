# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:24:09 2020

@author: KEPetousakis
"""

import numpy as np
import Code_General_utility_spikes_pickling as util
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy
import pandas as pd
import os
from copy import deepcopy as dcp
from sklearn.linear_model import LogisticRegression as LogisticRegression
from sklearn.preprocessing import MinMaxScaler as Normalize
from sklearn.linear_model import LinearRegression as LinearRegression
import scipy.stats as stats
import warnings
import logging

logging.getLogger('matplotlib.font_manager').disabled = True

warnings.filterwarnings("ignore")


font = {'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

__RES_DPI = 48

dt = 0.1
verbose = False

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
	# "ELP" in this code stands for "Electrotonic (Length) Constant"
	dendrite_information[segname] = {'diameter':info['diam'][idx],'length':info['L'][idx],'distance':info['totlength'][idx],'ELP':1/info['elp'][idx],'tELP':info['telp'][idx],'order':info['nc'][idx],'terminal':False}


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


def find_inflection(data, dt=0.1):
	
	data_corrected = [x+79 for x in data]
	
	data_corrected = np.array(data_corrected)
	
	#data.round(decimals=2)
	
	data = data_corrected[int(220/dt):]
	offset = int(220/dt)
	
	startat = util.Find(data==max(data))[0]+int(0.25/dt) # 0.25ms past max Vm
	stopat = len(data) #-(200/dt)
	
	
	der1 = np.gradient(data)
	der2 = np.gradient(der1)
	
	nIP = 0
	infl_points = []
	for i,point in enumerate(der2):
		if i >= startat and i < stopat and i < len(der2)-1:
			p1 = (i,point)
			p2 = (i+1, der2[i+1])
			if p1[1]*p2[1] < 0:
				nIP+=1
				# infl_points.append(i)
				infl_points.append(i+offset)
				
# 	print(f"Found {nIP} inflection points at {infl_points}")
	return (nIP, infl_points, data)


def nmda_detect(data):
	
	(nIP, infl_points, data_fixed) = find_inflection(data)
	
	if nIP >= 2 and max(data_fixed) >= -20+79:  # Careful: values near Vrest can behave chaotically for the second derivative, leading to extreme numbers of IPs
		return True
	else:
		return False
	
	
def width_halfmax_amp(data,naspike=0,dt=0.1):
	
	data_corrected = [x+79 for x in data]
	
	data = np.array(data_corrected)
	
	if naspike == 0:
		data = data[int(220/dt):]
		offset = int(220/dt)
	
	startat = util.Find(data==max(data))[0]+int(0.25/dt)  # 0.25ms past max Vm
	
	halfmax = data[startat]/2
	
	data_remainder = data[startat:]
	
	abs_minus_halfmax = [abs(x-halfmax) for x in data_remainder]
	
	projection = util.Find(abs_minus_halfmax == min(abs_minus_halfmax))[0]
	
	projection += startat
	
	width = projection - startat

	return width*dt # return value in ms


def trace_auc(data):
	
	data_corrected = [x+79 for x in data]
	
	data = np.array(data_corrected)
	
	startat = util.Find(data==max(data))[0]+int(0.25/dt)  # 0.25ms past max Vm
	
	auc = np.trapz(data[startat:])

	return auc


def calc_NRLE(dend_na_maxima):
	# Maximum ratio of (actual/expected) responses for a dendrite
	
	NRLE = 0
	
	dend_na_maxima +=79

	for nstim, maximum in enumerate(dend_na_maxima):
		if nstim > 1:
			# grad = np.polyfit(xax[0:nstim], dend_na_maxima[0:nstim], deg=1)[0]
			# grad = dend_na_maxima[0]
			grad = dend_na_maxima[1] - dend_na_maxima[0]
			extrap = lambda x : grad*(x+1)
			local_ratio = maximum/extrap(nstim)
			if local_ratio > NRLE:
				NRLE = local_ratio
	
	return NRLE
	

if __name__ == "__main__":
	
	
	if not os.path.exists('./Data/F1/Data_F1_F5_S1.pickle'):
		na_data = util.pickle_load('./Data/F1/Data_F1_F6_S1_Na_responses.pickle')
		nmda_data = util.pickle_load('./Data/F1/Data_F1_F6_S1_NMDA_noNa_responses.pickle')
		print('Loaded raw data...')
		process_traces = True
	else:
		(na_thresholds,nmda_thresholds,na_maxima,na_maxima_backup,nmda_na_maxima,dendrite_NRLEs,traces_to_visualize_basal,traces_to_visualize_apical) = util.pickle_load('./Data/F1/Data_F1_F5_S1.pickle')
		print('Loaded processed data...')
		process_traces = False
	
	
	dendrite_names = [f"basal{x}" for x in range(0,7)]
	for x in range(0,43):
		dendrite_names.append(f'apical{x}')
	
	if process_traces:
		print('Processing raw data...')
		# Sodium spiking threshold calculation and actual max response per dendrite
		na_thresholds = [np.nan for x in range(0,50)]
		for i,dendrite in enumerate(na_data):
			for nstim, trace in enumerate(dendrite):
				(spikes, _) = util.GetSpikes(trace[0:int(220/dt)], -20, False, 'max')
				if spikes > 0:
					na_thresholds[i] = nstim+1  #to avoid zeroes (off-by-one error)
					print(f'Compartment {dendrite_names[i]} has a Na+ spiking threshold of {na_thresholds[i]}')
					break
				if spikes == 0 and nstim == len(dendrite)-1:
					print(f'Compartment {dendrite_names[i]} did not exhibit Na+ spikes for up to 200 synapses.')
					
				
		# NMDA spiking threshold calculation and half max amplitude EPSP width per dendrite
		nmda_thresholds = [np.nan for x in range(0,50)]
		for i,dendrite in enumerate(nmda_data):
			for nstim, trace in enumerate(dendrite):
				spikes = nmda_detect(trace)
				if spikes:
					nmda_thresholds[i] = nstim+1  #to avoid zeroes (off-by-one error)
					print(f'Compartment {dendrite_names[i]} has an NMDA spiking threshold of {nmda_thresholds[i]}')
					break
				if not spikes and nstim == len(dendrite)-1:
					print(f'Compartment {dendrite_names[i]} did not exhibit NMDA spikes for up to 200 synapses.')
					
				
		na_maxima = np.zeros(shape=(50,200))
		print('Calculating Na+ maxima...')
		for i,dendrite in enumerate(na_data):
			for nstim, trace in enumerate(dendrite):
				na_maxima[i][nstim] = max(trace[0:int(220/dt)])
				
		na_maxima_backup = dcp(na_maxima)
				
		nmda_na_maxima = np.zeros(shape=(50,200))
		print('Calculating NMDA maxima...')
		for i,dendrite in enumerate(nmda_data):
			for nstim, trace in enumerate(dendrite):
				nmda_na_maxima[i][nstim] = max(trace[np.argmax(trace)+int(0.25/dt):]) # calculation: starting 0.25ms after the 2nd pulse peak (global peak)
					
		# NRLE value calculation per dendrite for sodium spikes, using max amplitude (Na+)
		print('Calculating NRLEs...')
		dendrite_NRLEs = [np.nan for x in range(0,50)]
		for i,dendrite in enumerate(na_maxima):
			(dendrite_NRLEs[i]) = calc_NRLE(dendrite)
			print(f'Dendrite {i} exhibits a Na+ NRLE of {dendrite_NRLEs[i]}')
			
		traces_to_visualize_basal = nmda_data[0]
		traces_to_visualize_apical = nmda_data[49]
			
		util.pickle_dump((na_thresholds,nmda_thresholds,na_maxima,na_maxima_backup,nmda_na_maxima,dendrite_NRLEs,traces_to_visualize_basal,traces_to_visualize_apical), './Data_F1_F6_S1_processed.pickle')
		
	
# PLOTS START HERE ======================================================================================================================	

#%% Expected vs Actual plot for basal dendrite 0 (Figure 1B)

# na_maxima = np.zeros(shape=(50,200))
# for i,dendrite in enumerate(na_data):
# 	for nstim, trace in enumerate(dendrite):
# 		na_maxima[i][nstim] = max(trace[0:int(220/dt)])

layout = [['.','B','C'],
	      ['D','E','F']]

fig, axes = plt.subplot_mosaic(layout, dpi = __RES_DPI)

na_maxima = dcp(na_maxima_backup)

xticks = [x for x in range(0,201)]
all_expected = []
# fig = plt.figure(dpi = __RES_DPI)
plt.sca(axes['B'])
for i, trace in enumerate([na_maxima[0]]):
	trace_fixed = [0]
	for element in trace:
		trace_fixed.append(element+79)
	expected_coeff = trace_fixed[1]
	expected = lambda x : expected_coeff*(x)
	xax = [expected(x) for x in xticks]
	all_expected.append(xax)
	plt.plot(xax,trace_fixed,'k', alpha=1)

expected_mean = np.min(all_expected,axis=0)
expected_std = np.std(all_expected, axis=0)
plt.plot(expected_mean, expected_mean, '--k', alpha=0.3)
plt.title('Figure 1B')
plt.xlabel('Expected peak ΔVm (mV)')
plt.ylabel('Actual peak ΔVm (mV)')
plt.xlim(0,100)
plt.ylim(0,120)
fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()


#%% Expected vs Actual plot for apical dendrite 0 (Figure 1C)

# na_maxima = np.zeros(shape=(50,200))
# for i,dendrite in enumerate(na_data):
# 	for nstim, trace in enumerate(dendrite):
# 		na_maxima[i][nstim] = max(trace[0:int(220/dt)])

na_maxima = dcp(na_maxima_backup)

xticks = [x for x in range(0,201)]
all_expected = []
# fig = plt.figure(dpi = __RES_DPI)
plt.sca(axes['C'])
for i, trace in enumerate([na_maxima[7]]):
	trace_fixed = [0]
	for element in trace:
		trace_fixed.append(element+79)
	expected_coeff = trace_fixed[1]
	expected = lambda x : expected_coeff*(x)
	xax = [expected(x) for x in xticks]
	all_expected.append(xax)
	plt.plot(xax,trace_fixed,'r', alpha=1)

expected_mean = np.min(all_expected,axis=0)
expected_std = np.std(all_expected, axis=0)
plt.plot(expected_mean, expected_mean, '--k', alpha=0.3)
plt.title('Figure 1C')
plt.xlabel('Expected peak ΔVm (mV)')
plt.ylabel('Actual peak ΔVm (mV)')
plt.xlim(0,100)
plt.ylim(0,120)
fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

#%% Histogram of apical and basal NRLEs  (Figure 1D)
NRLE_bins = [0.1*x for x in range(0,int(4/0.1)+1)] # 0-4 with a step of 0.05
A_NRLEs = dendrite_NRLEs[7:]
B_NRLEs = dendrite_NRLEs[0:7]

(muA, sigmaA) = norm.fit(A_NRLEs)
(muB, sigmaB) = norm.fit(B_NRLEs)

# fig = plt.figure(dpi = __RES_DPI)
plt.sca(axes['D'])

(nA, binsA, patchesA) = plt.hist(A_NRLEs, NRLE_bins, color='r', alpha=0.5)
(nB, binsB, patchesB) = plt.hist(B_NRLEs, NRLE_bins, color='k', alpha=0.5)


plt.legend(['Apical Na+ NRLEs','Basal Na+ NRLEs']); plt.xlabel('NRLE values'); plt.ylabel('Number of dendrites')
plt.title('Figure 1D')

fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.xlim((0.5,4))


#%% Sorted bar plots of spiking thresholds, one for apical and one for basal dendrites (Figure 1E / Figure 1F)
naTA = na_thresholds[7:]
naTA_idx = np.argsort(naTA)
naTA = np.sort(naTA)
np.flip(naTA)

naTB = na_thresholds[0:7]
naTB_idx = np.argsort(naTB)
naTB = np.sort(naTB)
np.flip(naTB)

nmdaTA = nmda_thresholds[7:]
nmdaTA = [nmdaTA[x] for x in naTA_idx]

nmdaTB = nmda_thresholds[0:7]
nmdaTB = [nmdaTB[x] for x in naTB_idx]

xA = [x for x in range(1,44)]
xB = [x for x in range(1,8)]

# fig = plt.figure(dpi = __RES_DPI)
plt.sca(axes['E'])
plt.bar(xA, naTA, width=-0.45, color='b', align='edge'); plt.bar(xA, nmdaTA, width=0.45, color='r', align='edge'); plt.ylim([0,110]); fig.set_tight_layout(True)
plt.legend(['Sodium spike threshold','NMDA spike threshold'])
plt.title('Figure 1E'); plt.xlabel('Apical dendrites'); plt.ylabel("Threshold (# synapses)"); plt.xticks([])
fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# fig = plt.figure(dpi = __RES_DPI)
plt.sca(axes['F'])
plt.bar(xB, naTB, width=-0.45, color='b', align='edge'); plt.bar(xB, nmdaTB, width=0.45, color='r', align='edge'); plt.ylim([0,110]); fig.set_tight_layout(True)
plt.legend(['Sodium spike threshold','NMDA spike threshold'])
plt.title('Figure 1F'); plt.xlabel('Basal dendrites'); plt.ylabel("Threshold (# synapses)"); plt.xticks([])
fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
	
	
#%% Expected vs actual mass plots for Na+ spikes, for basal dendrites (Supp.Figure 1A)

layout = [['A','B'],
	      ['C','D']]

fig, axes = plt.subplot_mosaic(layout, dpi = __RES_DPI)

na_maxima = dcp(na_maxima_backup)

xticks = [x for x in range(0,201)]
all_expected = []
# fig = plt.figure(dpi = __RES_DPI)
plt.sca(axes['A'])
for i, trace in enumerate(na_maxima[0:7]):
	trace_fixed = [0]
	for element in trace:
		trace_fixed.append(element+79)
	expected_coeff = trace_fixed[1]
	expected = lambda x : expected_coeff*(x)
	xax = [expected(x) for x in xticks]
	all_expected.append(xax)
	if i == 0:
		plt.plot(xax,trace_fixed,'b', alpha=1)
	else:
		plt.plot(xax,trace_fixed,'k', alpha=0.3)

expected_mean = np.min(all_expected,axis=0)
expected_std = np.std(all_expected, axis=0)
plt.plot(expected_mean, expected_mean, '--r')
plt.title('Supp.Figure 1A')
plt.xlabel('Expected peak dVm')
plt.ylabel('Actual peak dVm')
plt.xlim(0,100)
plt.ylim(0,120)
fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

#%% Expected vs actual mass plots for Na+ spikes, for apical dendrites (Supp.Figure 1B)
na_maxima = dcp(na_maxima_backup)

highlight = 0
if highlight != 0:
	form = '.--'
else:
	form = ''
xticks = [x for x in range(0,201)]
# fig = plt.figure(dpi = __RES_DPI)
plt.sca(axes['B'])
for i, trace in enumerate(na_maxima[7:]):
	trace_fixed = [0]
	for element in trace:
		trace_fixed.append(element+79)
	expected_coeff = trace_fixed[1]
	expected = lambda x : expected_coeff*(x)
	xax = [expected(x) for x in xticks]
	if i != highlight:
		plt.plot(xax,trace_fixed,'k', alpha=0.3)
	else:
		plt.plot(xax,trace_fixed,f'{form}b', alpha=1)

expected_mean = np.min(all_expected,axis=0)
expected_std = np.std(all_expected, axis=0)
plt.plot(expected_mean, expected_mean, '--r')
plt.title('Supp.Figure 1B')
plt.xlabel('Expected peak dVm')
plt.ylabel('Actual peak dVm')
plt.xlim(0,100)
plt.ylim(0,120)
fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()


#%% Traces from basal dendrite 0 (Supplementary Figure 1C)
if process_traces:
	visual_selected_compartment = 0
	vsc = visual_selected_compartment

# fig = plt.figure(dpi = __RES_DPI)
plt.sca(axes['C'])
opacity = [(200-x)/200 for x in range(0,200)]; opacity.reverse()
xax = [(x*dt)-200 for x in range(0,int(500/dt+1))]
syn_range = [x*10 for x in range(0,21)]
legend_str = []
for isyn in range(1,201):
	if isyn in syn_range:
		c='k'
		if process_traces:
			plt.plot(xax, nmda_data[vsc][isyn-1][:], c , alpha = opacity[isyn-1]) 
		else:
			plt.plot(xax, traces_to_visualize_basal[isyn-1][:], c , alpha = opacity[isyn-1]) 
		legend_str.append(f'nsyn={isyn}')
plt.xlabel('Time since stimulus onset (ms)')
plt.ylabel('Compartment Vm (mV)')
plt.xlim([-20, 150])
# plt.legend(legend_str)
plt.title('Supp.Figure 1C')
fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()


#%% Traces from apical dendrite 42 (Supplementary Figure 1D)
if process_traces:
	visual_selected_compartment = 49
	vsc = visual_selected_compartment

# fig = plt.figure(dpi = __RES_DPI)
plt.sca(axes['D'])
opacity = [(200-x)/200 for x in range(0,200)]; opacity.reverse()
xax = [(x*dt)-200 for x in range(0,int(500/dt+1))]
syn_range = [x*10 for x in range(0,21)]
legend_str = []
for isyn in range(1,201):
	if isyn in syn_range:
		c='r'
		if process_traces:
			plt.plot(xax, nmda_data[vsc][isyn-1][:], c , alpha = opacity[isyn-1]) 
		else:
			plt.plot(xax, traces_to_visualize_apical[isyn-1][:], c , alpha = opacity[isyn-1]) 
		legend_str.append(f'nsyn={isyn}')
plt.xlabel('Time since stimulus onset (ms)')
plt.ylabel('Compartment Vm (mV)')
plt.xlim([-20, 150])
# plt.legend(legend_str)
plt.title('Supp.Figure 1D')
fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.show()
