import Code_General_utility_spikes_pickling as util
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dcp
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

_REMOVE_BAPS = True

# For converted model data
frame_time = 0.138716725543428 #seconds
max_dt = 50  # Original: 10
bin_size = 1 # Original: 1

_NEURONS = [x for x in range(0,50)]
_RUNS = [x for x in range(0,100)]

try:
	distr_all_d = util.pickle_load('./Data/S2/Data_F2F_Model_Distribution.pickle')
except:
	raise Exception

distr_dict = dcp(distr_all_d)
neg_nozero = [distr_all_d[f'{x}'] for x in range(-50,0)] # change to range(-50,-4) to check percentage without the 4 bins nearest to 0
neg_nozero_no4 = [distr_all_d[f'{x}'] for x in range(-50,-4)]
sum_neg_nozero = sum(neg_nozero)
sum_neg_nozero_no4 = sum(neg_nozero_no4)
pos_nozero = [distr_all_d[f'{x}'] for x in range(1,51)]
sum_pos_nozero = sum(pos_nozero)
distr_all_d = [distr_all_d[f'{x}'] for x in range(-50,51)]
all_spikes = sum(distr_all_d)
all_spikes_nozero = all_spikes-max(distr_all_d)

print(f'All spikes: {all_spikes}\nWithout BAPs: {all_spikes_nozero}\n Dendrite first percentage (no BAPs): {sum_neg_nozero/all_spikes_nozero}\n Soma first percentage (no BAPs): {sum_pos_nozero/all_spikes_nozero}')
print(f'Dendrite first percentage (no BAPs, excluding 4 closest frames): {sum_neg_nozero_no4/all_spikes_nozero}')

bap_index = np.argmax(distr_all_d)
print(f'BAPs detected for index {bap_index}')

print(f'Without removing BAPs: \n Dendrite first percentage: {sum_neg_nozero/all_spikes}\n Soma first percentage: {sum_pos_nozero/all_spikes}')


a = [i for i in range(0, 2*int(max_dt/bin_size))]
x_d = [(i-int(len(a)/2))*bin_size*frame_time for i in range(0, 2*int(max_dt/bin_size))]

distr_all_d_new = []

if _REMOVE_BAPS:
	x_d_new = [y for x,y in enumerate(x_d) if x != bap_index]  # to remove BAP index bin
	effective_max_spikes = sum(distr_all_d) - max(distr_all_d)  # to remove BAP index bin
	# Removing BAP index from bins
	for x,y in enumerate(x_d):
		if x != bap_index:
			distr_all_d_new.append(distr_all_d[x]/effective_max_spikes)
else:
	x_d_new = [y for x,y in enumerate(x_d)]
	effective_max_spikes = sum(distr_all_d)
	for x,y in enumerate(x_d):
		distr_all_d_new.append(distr_all_d[x]/effective_max_spikes)


# Figure 2F (uncolored)
fig = plt.figure(dpi = __RES_DPI)
plt.bar(x_d_new,distr_all_d_new, width = 0.1)
if frame_time > 0.1:
	plt.xlabel('Time distance of events (s)')
else:
	plt.xlabel('Time distance of events (ms)')
	plt.xticks(x_d_new)
plt.ylabel('Fraction of event pairs')
plt.title('Figure 2F (uncolored)')
plt.ylim(0,0.12)

fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.show()
