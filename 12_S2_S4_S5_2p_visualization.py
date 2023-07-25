# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 15:16:58 2022

@author: KEPetousakis
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import Code_General_utility_spikes_pickling as util
import matplotlib as mpl
import warnings
import logging

logging.getLogger('matplotlib.font_manager').disabled = True

warnings.filterwarnings("ignore")

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

mpl.rc('font', **font)

__RES_DPI = 48*0.75

def count_cases(bins, labels, condition):
	"""Given a set of bins, and their labels, counts number (and percentage) of cases (bins) whose labels satisfy a condition."""

	sum_cases = 0
	try:
		for i,label in enumerate(labels):		
			if eval(f'{label}{condition}'):  # careful with this, can throw errors like no-one's business.
				sum_cases+=bins[i]
		percentage_cases = sum_cases/sum(bins)
		print(f'\tFound {sum_cases} fitting condition "Δt{condition} s" ({percentage_cases*100}% of cases).')
		return(sum_cases, percentage_cases)

	except Exception as e:
		print(f'\tCaution! Odd condition entry to "count_cases()".\n{e}')

	return -1


if os.path.exists(f'./Data/S2/Data_F2_bins_2p.pickle') and os.path.exists(f'./Data/S2/Data_F2_distribution_2p.pickle'):
	x_d = util.pickle_load(f'./Data/S2/Data_F2_bins_2p.pickle')
	distr_all_d = util.pickle_load(f'./Data/S2/Data_F2_distribution_2p.pickle')
else:
	raise IOError

# For count_cases
max_dt = 36    # for tstop = 5 sec
bin_size = 1 # Original: 1
frame_time = 0.138716725543428
a = [i for i in range(0, 2*int(max_dt/bin_size))]
x_d = [(i-int(len(a)/2))*bin_size*frame_time for i in range(1, 2*int(max_dt/bin_size))]
x_d = [x*frame_time for x in range(-max_dt, max_dt+1)]

print(f'Total cases of Tdend(i)-Tsoma: {int(sum(distr_all_d))} ({int(max(distr_all_d))}/{int(sum(distr_all_d))} presumed BAPs)')
print(f'\t[Soma first] Positive percentage of Tdend(i)-Tsoma: {sum([x for x,y in zip(distr_all_d,x_d) if y>0])/sum(distr_all_d)}')
print(f'\t[Even timing] Zeroes percentage of Tdend(i)-Tsoma: {sum([x for x,y in zip(distr_all_d,x_d) if y==0])/sum(distr_all_d)}')
print(f'\t[Dendrite first] Negative percentage of Tdend(i)-Tsoma: {sum([x for x,y in zip(distr_all_d,x_d) if y<0])/sum(distr_all_d)}')
print(f'\t[Dendrite first, even timings out] Positive percentage of Tdend(i)-Tsoma: {sum([x for x,y in zip(distr_all_d,x_d) if y<0])/(sum(distr_all_d)-sum([x for x,y in zip(distr_all_d,x_d) if y==0]))}')
# This calculates the percentage of dfirst without the 4 bins closest to zero
print(f'\t[Dendrite first, even timings out] Positive percentage of Tdend(i)-Tsoma (excluding 4 bins closest to 0): {sum([x for x,y in zip(distr_all_d,x_d) if y<-(4/7.25)])/(sum(distr_all_d)-sum([x for x,y in zip(distr_all_d,x_d) if y==0]))}')
print(f'\t[Soma first, even timings out] Negative percentage of Tdend(i)-Tsoma: {sum([x for x,y in zip(distr_all_d,x_d) if y>0])/(sum(distr_all_d)-sum([x for x,y in zip(distr_all_d,x_d) if y==0]))}')
count_cases(distr_all_d, x_d, '>0.5')
count_cases(distr_all_d, x_d, '<-0.5')
count_cases(distr_all_d, x_d, '<-0.138')  # Percentage of dendritic spikes without a consequent somatic spike within 0.139 seconds (~1 frame)


layout = [[x for x in '.AAAAA.'],[x for x in 'BBB....']]

fig_main, axes_main = plt.subplot_mosaic(layout, dpi = __RES_DPI)

#%% Figure 2E (not including the bin at x=0; uncolored)
remove_bap = True

# fig = plt.figure(dpi = __RES_DPI)
plt.sca(axes_main['B'])
if remove_bap:
	x_d_new = [x for x in x_d if x != 0]  # to remove 0-bin
	effective_max_spikes = sum(distr_all_d) - max(distr_all_d) # to remove 0-bin
	distr_all_d_new = [distr_all_d[x]/effective_max_spikes for x,y in enumerate(x_d) if y != 0]  # to remove 0-bin
else:
	x_d_new = [x for x in x_d]
	effective_max_spikes = sum(distr_all_d)
	distr_all_d_new = [distr_all_d[x]/effective_max_spikes for x,y in enumerate(x_d)]
	
plt.bar(x_d_new,distr_all_d_new, width = 0.1)
plt.title('Supp.Figure 2E (uncolored)')
plt.xlabel('Time distance of events (s)')
plt.ylabel('Fraction of event pairs')
plt.ylim((0,0.12))
plt.xlim(-(frame_time*50),(frame_time*50))
fig_main.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()


#%% Figure 2D
[aggregate_somata, aggregate_dendrites, averaged_somata, averaged_dendrites, n_somata, n_dendrites, err_somata, err_dendrites] = util.pickle_load('./Data/S2/Data_F2_allcells_STA.pickle')
window_left = 5
window_right = 10
window_full = window_left+window_right+1

# fig = plt.figure(dpi = __RES_DPI)
plt.sca(axes_main['A'])
plt.errorbar(range(-window_left, window_full-window_left), np.squeeze(averaged_somata), yerr=err_somata, c='b', label='Somatic STA', capsize=10, elinewidth=1)
plt.errorbar(range(-window_left, window_full-window_left), np.squeeze(averaged_dendrites), yerr=err_dendrites, c='r', label='Dendritic STA', capsize=10, elinewidth=1)
plt.axvline(0, ls='--', c='k')
plt.title('Supp.Figure 2D')
plt.xlabel('Number of dt since event onset')
plt.ylabel('Adjusted fluorescence intensity (ΔF/F0)')
plt.legend()
fig_main.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()


#%% Supp.Figure 4A
layout = [['.','A1','.'],['.','A2','.'],['.','B1','.'],['.','B2','.'],['.','C1','.'],['.','C2','.']]
iA = ['A1','A2']; iB = ['B1','B2']; iC = ['C1','C2']

fig, axes = plt.subplot_mosaic(layout, dpi = __RES_DPI)

[aggregate_somata, aggregate_dendrites, averaged_somata, averaged_dendrites, n_somata, n_dendrites, err_somata, err_dendrites] = util.pickle_load('./Data/S2/Data_F2_allcells_STA.pickle')
plt.sca(axes['A1'])
# fig = plt.figure(dpi = __RES_DPI)
plt.suptitle('Supp.Figure 4')
# plt.subplot(211)
for i in range(0,len(aggregate_somata)):
	plt.plot(aggregate_somata[i][:], c='b', alpha=i/len(aggregate_somata))
plt.title('Somatic events')
plt.xlabel('Time (dt)')
plt.ylabel('Adjusted fluorescence intensity (ΔF/F0)', fontsize=12)

	
# plt.subplot(212)
plt.sca(axes['A2'])
for i in range(0,len(aggregate_dendrites)):
	plt.plot(aggregate_dendrites[i][:], c='r', alpha=i/len(aggregate_dendrites))
plt.title('Dendritic events')
plt.xlabel('Time (dt)')
plt.ylabel('Adjusted fluorescence intensity (ΔF/F0)', fontsize=12)


fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
			

#%% Supp.Figure 4B
[aggregate_somata, aggregate_dendrites, averaged_somata, averaged_dendrites, n_somata, n_dendrites, err_somata, err_dendrites] = util.pickle_load('./Data/S2/Data_F2_mouse1_STA.pickle')
plt.sca(axes['B1'])
# fig = plt.figure(dpi = __RES_DPI)
# plt.suptitle('Supp.Figure 4B')
# plt.subplot(211)
for i in range(0,len(aggregate_somata)):
	plt.plot(aggregate_somata[i][:], c='b', alpha=i/len(aggregate_somata))
plt.title('Somatic events')
plt.xlabel('Time (dt)')
plt.ylabel('Adjusted fluorescence intensity (ΔF/F0)', fontsize=12)

	
# plt.subplot(212)
plt.sca(axes['B2'])
for i in range(0,len(aggregate_dendrites)):
	plt.plot(aggregate_dendrites[i][:], c='r', alpha=i/len(aggregate_dendrites))
plt.title('Dendritic events')
plt.xlabel('Time (dt)')
plt.ylabel('Adjusted fluorescence intensity (ΔF/F0)', fontsize=12)


fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()


#%% Supp.Figure 4C
[aggregate_somata, aggregate_dendrites, averaged_somata, averaged_dendrites, n_somata, n_dendrites, err_somata, err_dendrites] = util.pickle_load('./Data/S2/Data_F2_mouse2_STA.pickle')
plt.sca(axes['C1'])
# fig = plt.figure(dpi = __RES_DPI)
# plt.suptitle('Supp.Figure 4C')
# plt.subplot(211)
for i in range(0,len(aggregate_somata)):
	plt.plot(aggregate_somata[i][:], c='b', alpha=i/len(aggregate_somata))
plt.title('Somatic events')
plt.xlabel('Time (dt)')
plt.ylabel('Adjusted fluorescence intensity (ΔF/F0)', fontsize=12)

	
# plt.subplot(212)
plt.sca(axes['C2'])
for i in range(0,len(aggregate_dendrites)):
	plt.plot(aggregate_dendrites[i][:], c='r', alpha=i/len(aggregate_dendrites))
plt.title('Dendritic events')
plt.xlabel('Time (dt)')
plt.ylabel('Adjusted fluorescence intensity (ΔF/F0)', fontsize=12)


fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()


#%% Supp.Figure 5A

layout = [[x for x in '.AAA.'],[x for x in '.BBB.'],[x for x in '.CCC.']]

fig, axes = plt.subplot_mosaic(layout, dpi = __RES_DPI)

[aggregate_somata, aggregate_dendrites, averaged_somata, averaged_dendrites, n_somata, n_dendrites, err_somata, err_dendrites] = util.pickle_load('./Data/S2/Data_F2_mouse1_STA.pickle')
window_left = 5
window_right = 10
window_full = window_left+window_right+1

plt.sca(axes['A'])
# fig = plt.figure(dpi = __RES_DPI)
plt.errorbar(range(-window_left, window_full-window_left), np.squeeze(averaged_somata), yerr=err_somata, c='b', label='Somatic STA', capsize=10, elinewidth=1)
plt.errorbar(range(-window_left, window_full-window_left), np.squeeze(averaged_dendrites), yerr=err_dendrites, c='r', label='Dendritic STA', capsize=10, elinewidth=1)
plt.axvline(0, ls='--', c='k')
plt.title('Supp.Figure 5A')
plt.xlabel('Number of dt since event onset')
plt.ylabel('Adjusted fluorescence intensity (ΔF/F0)')
plt.legend()
fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()


#%% Supp.Figure 5B
[aggregate_somata, aggregate_dendrites, averaged_somata, averaged_dendrites, n_somata, n_dendrites, err_somata, err_dendrites] = util.pickle_load('./Data/S2/Data_F2_mouse2_STA.pickle')
window_left = 5
window_right = 10
window_full = window_left+window_right+1

plt.sca(axes['B'])
# fig = plt.figure(dpi = __RES_DPI)
plt.errorbar(range(-window_left, window_full-window_left), np.squeeze(averaged_somata), yerr=err_somata, c='b', label='Somatic STA', capsize=10, elinewidth=1)
plt.errorbar(range(-window_left, window_full-window_left), np.squeeze(averaged_dendrites), yerr=err_dendrites, c='r', label='Dendritic STA', capsize=10, elinewidth=1)
plt.axvline(0, ls='--', c='k')
plt.title('Supp.Figure 5B')
plt.xlabel('Number of dt since event onset')
plt.ylabel('Adjusted fluorescence intensity (ΔF/F0)')
plt.legend()
fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()


#%% Supp.Figure 5C

if os.path.exists(f'./Data/S2/Data_F2_bins_2p.pickle') and os.path.exists(f'./Data/S2/Data_F2_distribution_2p.pickle'):
	x_d = util.pickle_load(f'./Data/S2/Data_F2_bins_2p.pickle')
	distr_all_d = util.pickle_load(f'./Data/S2/Data_F2_distribution_2p.pickle')
else:
	raise IOError

# For count_cases
max_dt = 36    # for tstop = 5 sec
bin_size = 1 # Original: 1
frame_time = 0.138716725543428
a = [i for i in range(0, 2*int(max_dt/bin_size))]
x_d = [(i-int(len(a)/2))*bin_size*frame_time for i in range(1, 2*int(max_dt/bin_size))]
x_d = [x*frame_time for x in range(-max_dt, max_dt+1)]

remove_bap = False

plt.sca(axes['C'])

if remove_bap:
	x_d_new = [x for x in x_d if x != 0]  # to remove 0-bin
	effective_max_spikes = sum(distr_all_d) - max(distr_all_d) # to remove 0-bin
	distr_all_d_new = [distr_all_d[x]/effective_max_spikes for x,y in enumerate(x_d) if y != 0]  # to remove 0-bin
else:
	x_d_new = [x for x in x_d]
	effective_max_spikes = sum(distr_all_d)
	distr_all_d_new = [distr_all_d[x]/effective_max_spikes for x,y in enumerate(x_d)]
	
plt.bar(x_d_new,distr_all_d_new, width = 0.1)
plt.title('Supp.Figure 5C')
plt.xlabel('Time distance of events (s)')
plt.ylabel('Fraction of event pairs')
plt.ylim((0,0.12))
plt.xlim(-(frame_time*50),(frame_time*50))
fig_main.set_tight_layout(True)


if os.path.exists(f'./Data/S4/Data_S4_bins_null.pickle') and os.path.exists(f'./Data/S4/Data_S4_distribution_null.pickle'):
	x_d = util.pickle_load(f'./Data/S4/Data_S4_bins_null.pickle')
	distr_all_d = util.pickle_load(f'./Data/S4/Data_S4_distribution_null.pickle')
else:
	raise IOError

# For count_cases
max_dt = 36    # for tstop = 5 sec
bin_size = 1 # Original: 1
frame_time = 0.138716725543428
a = [i for i in range(0, 2*int(max_dt/bin_size))]
x_d = [(i-int(len(a)/2))*bin_size*frame_time for i in range(1, 2*int(max_dt/bin_size))]
x_d = [x*frame_time for x in range(-max_dt, max_dt+1)]


remove_bap = False


if remove_bap:
	x_d_new = [x for x in x_d if x != 0]  # to remove 0-bin
	effective_max_spikes = sum(distr_all_d) - max(distr_all_d) # to remove 0-bin
	distr_all_d_new = [distr_all_d[x]/effective_max_spikes for x,y in enumerate(x_d) if y != 0]  # to remove 0-bin
else:
	x_d_new = [x for x in x_d]
	effective_max_spikes = sum(distr_all_d)
	distr_all_d_new = [distr_all_d[x]/effective_max_spikes for x,y in enumerate(x_d)]
	
plt.bar(x_d_new,distr_all_d_new, width = 0.1, alpha=0.5)
plt.xlabel('Time distance of events (s)')
plt.ylabel('Fraction of event pairs')
plt.ylim((0,0.12))
plt.xlim(-(frame_time*50),(frame_time*50))
fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()