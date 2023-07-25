import numpy as np
import Code_General_utility_spikes_pickling as util
from copy import deepcopy as dcp
import matplotlib.pyplot as plt
import Code_General_Nassi_functions as nf

__RES_DPI = 48

_CND = 600
_THRESHOLD = -20 
_DISP = 0
_N_NEURONS = 10
_N_RUNS = 10
_N_STIMULI = 18
_T_STIM = 2 # s
_T_STIM_PRESENT = 500 # ms
_DT = 0.1 # ms
_CONDITIONS = {200:"20:80",300:"30:70",400:"40:60",500:"50:50", 600:"60:40",700:"70:30",800:"80:20"}

_OVERWRITE = False
_ONE_NEURON = False
_ALL_NEURONS = True
_MULTIPLE_NEURONS = False

_TRANSFORM_VECTOR = np.array([9,8,7,6,5,4,3,2,1,0,17,16,15,14,13,12,11,10,9])
_TRANSFORM_VECTOR = np.flip(_TRANSFORM_VECTOR)

neurons = [x for x in range(0,_N_NEURONS)]
runs = [y for y in range(0,_N_RUNS)]
stims = [z*10 for z in range(0,_N_STIMULI)]


# Apical trunk ablation
_FILEPATH = './Data/S9'
affix = '_ablated'
normalize = True

# Control, to compare against apical trunk ablation
_FILEPATH2 = './Data/S9/firing_rates_stds_cnd600.pickle'


firing_rates_all_neurons_all_runs = np.zeros(shape = (_N_NEURONS,_N_RUNS,_N_STIMULI))
firing_rates_all_neurons_across_runs = np.zeros(shape = (_N_NEURONS,_N_STIMULI))
firing_rates_across_neurons_across_runs = np.zeros(shape = (_N_STIMULI))

stds_all_neurons_all_runs = np.zeros(shape = (_N_NEURONS,_N_RUNS,_N_STIMULI))
stds_all_neurons_across_runs = np.zeros(shape = (_N_NEURONS,_N_STIMULI))
stds_across_neurons_across_runs = np.zeros(shape = (_N_STIMULI))

n_rates = []
n_errors = []
a_rates = []
a_errors = []

try:
	(n_rates,n_errors,a_rates,a_errors,firing_rates_all_neurons_all_runs) = util.pickle_load(f'{_FILEPATH}/intv_control_cnd{_CND}{affix}.pickle')

	if _OVERWRITE:
		raise Exception
	print('Processed data found, loading pickle...')
except:
	print(f'Did not find processed data. Processing data for disparity {_DISP} now...\n')
	raise Exception('Data processing pipeline is inoperative - processed data should be under ./data/S9/')

finally:
	print('=== RUN-WISE ===')
	global_rejections = 0
	for idx_n, nrn in enumerate(neurons):
		local_rejections = 0
		print(f'Neuron {nrn}')
		for idx_r, run in enumerate(runs):
			print(f'\tRun {run}', end = '  |  ')
			relevant_rates = np.squeeze(firing_rates_all_neurons_all_runs[idx_n,idx_r,:])
			(nrn_pref, nrn_OSI, nrn_width, _ , _ ) = nf.tuning_properties(relevant_rates, [x*10 for x in range(0,18)])
			if nrn_OSI < 0.2 or nrn_width > 80 or np.isnan(nrn_OSI):
				verdict = 'REJECT'
				local_rejections += 1
			else:
				verdict = 'ACCEPT'
			print(f'Pref {nrn_pref} OSI {nrn_OSI:.3f} width {nrn_width} {verdict}')
		if local_rejections > 2:
			print(f'\t <!> Rejecting neuron {nrn} for {local_rejections}/10 run rejections.')
			global_rejections += 1
	if global_rejections > 2:
		print(f'<!> Rejecting all neurons ({global_rejections}/10 neurons rejected).')
	else:
		print('<!> Neuron shows normal orientation tuning overall.')


if _ALL_NEURONS:

	print(f'Plotting tuning curve for disparity {_DISP}...')

	fig = plt.figure(dpi = __RES_DPI)
	fig.set_tight_layout(True)

	if len(a_rates) == 0 or len(a_errors) == 0:
		a_rates = firing_rates_across_neurons_across_runs[:]
		a_errors = stds_across_neurons_across_runs[:]
	else:
		a_rates = a_rates[:]
		a_errors = a_errors[:]
		

	x_axis = np.array([x*10 for x in range(-9,10)])
	y_axis = np.array([x for x in a_rates])
	y_errors = np.array([x/np.sqrt(10) for x in a_errors])
	
	if normalize:
		y_axis /= np.max(y_axis)
		y_errors /= np.max(y_axis)

	preference = np.argmax(y_axis)*10

	(nrn_pref, nrn_OSI, nrn_width, _ , _ ) = nf.tuning_properties(y_axis, [x*10 for x in range(0,18)])

	print(f'OSI/width analysis: preferred {nrn_pref} deg, OSI {nrn_OSI}, width {nrn_width}')

	print(f'Mean preferred orientation for disparity {_DISP}, condition {_CONDITIONS[_CND]}: {preference}')
	
	plt.errorbar(x_axis, y_axis[_TRANSFORM_VECTOR], y_errors[_TRANSFORM_VECTOR], c='r', capsize=4)
	plt.xticks(x_axis)
	# plt.title(f'Pref {nrn_pref}, OSI {nrn_OSI:.5f}, width {nrn_width}, condition "{affix}"')
	plt.title('Supplementary Figure 9')
	plt.ylim([0,2.2])
	if normalize:
		plt.ylim([0,1.2])
		
		(n_rates2,n_errors2,a_rates2,a_errors2,firing_rates_all_neurons_all_runs2) = util.pickle_load(f'{_FILEPATH2}')
		
		x_axis2 = np.array([x*10 for x in range(-9,10)])
		y_axis2 = np.array([x for x in a_rates2])
		y_errors2 = np.array([x/np.sqrt(10) for x in a_errors2])
		
		(nrn_pref2, nrn_OSI2, nrn_width2, _ , _ ) = nf.tuning_properties(y_axis2, [x*10 for x in range(0,18)])

		print(f'OSI/width analysis: preferred {nrn_pref2} deg, OSI {nrn_OSI2}, width {nrn_width2}')
		
		y_axis2 /= np.max(y_axis2)
		y_errors2 /= np.max(y_axis2)
		
		plt.errorbar(x_axis2, y_axis2[_TRANSFORM_VECTOR], y_errors2[_TRANSFORM_VECTOR], capsize=4)
		
	plt.xlabel('Stimulus orientation ($^\circ$)')
	plt.ylabel('Neuronal response (Hz)')
	# plt.legend([f'Preference: {preference}'])
	figmanager = plt.get_current_fig_manager()
	figmanager.window.showMaximized()
	plt.show()

