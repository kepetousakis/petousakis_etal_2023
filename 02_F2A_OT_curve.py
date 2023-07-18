import numpy as np
import Code_General_utility_spikes_pickling as util
from copy import deepcopy as dcp
import matplotlib.pyplot as plt
import Code_General_Nassi_functions as nf
import matplotlib as mpl

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

mpl.rc('font', **font)

__RES_DPI = 48

_CND = 600
_DISP = 0
_N_NEURONS = 10
_N_RUNS = 10
_N_STIMULI = 18
_CONDITIONS = {200:"20:80",300:"30:70",400:"40:60",500:"50:50", 600:"60:40",700:"70:30",800:"80:20"}

_OVERWRITE = False
_ALL_NEURONS = True

_TRANSFORM_VECTOR = np.array([9,8,7,6,5,4,3,2,1,0,17,16,15,14,13,12,11,10,9])
_TRANSFORM_VECTOR = np.flip(_TRANSFORM_VECTOR)

neurons = [x for x in range(0,_N_NEURONS)]
runs = [y for y in range(0,_N_RUNS)]
stims = [z*10 for z in range(0,_N_STIMULI)]

disp = 0

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
	(n_rates,n_errors,a_rates,a_errors,firing_rates_all_neurons_all_runs) = util.pickle_load(f'./Data/F2/firing_rates_stds_cnd{_CND}.pickle')
	
	# Ablated case
	# (n_rates,n_errors,a_rates,a_errors,firing_rates_all_neurons_all_runs) = util.pickle_load('/home/cluster/kostasp/l3v1models/currentmodel/results/revisions/Ablation_Tuning_Test_weaksyns/analysis/intv_control_cnd600_ablated.pickle')

	if _OVERWRITE:
		raise Exception
	print('Processed data found, loading pickle...')
except:
	print(f'Did not find processed data. Processing data for disparity {_DISP} now...\n')
	raise Exception('Processed data should be present - data processing pipeline is not functional.')

finally:
	
	prefs = []
	OSIs = []
	widths = []
	rates_pref = []
	rates_orth = []
	
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

	print('\n=== Evaluation ===')
	global_rejections = 0
	for idx_n, nrn in enumerate(neurons):
		print(f'Neuron {nrn}', end = '  |  ')
		relevant_rates = np.squeeze(firing_rates_all_neurons_across_runs[idx_n,:])
		if not _OVERWRITE and len(n_rates)>1:
			relevant_rates = np.squeeze(n_rates[idx_n,:])
		(nrn_pref, nrn_OSI, nrn_width, _ , _ ) = nf.tuning_properties(relevant_rates, [x*10 for x in range(0,18)])
		prefs.append(nrn_pref)
		OSIs.append(nrn_OSI)
		widths.append(nrn_width)
		rates_pref.append(relevant_rates[(nrn_pref//10)])
		if nrn_pref <= 90:
			rates_orth.append(relevant_rates[(nrn_pref//10)+9])
		else:
			rates_orth.append(relevant_rates[(nrn_pref//10)-9])

		if nrn_OSI < 0.2 or nrn_width > 80 or np.isnan(nrn_OSI):
			verdict = 'REJECT'
			global_rejections += 1
		else:
			verdict = 'ACCEPT'
		print(f'Pref {nrn_pref} OSI {nrn_OSI:.3f} width {nrn_width} \t{verdict}')
		if verdict == 'REJECT':
			print(f'\t\t\t', end='')
			for x in relevant_rates:
				print(f'{x:.04}', end=', ')
			print()
	if global_rejections > 2:
		print(f'<!> Rejecting all neurons ({global_rejections}/10 neurons rejected).')
	else:
		print('<!> Neuron shows normal orientation tuning overall.')


if _ALL_NEURONS:

	print(f'Plotting tuning curve for disparity {_DISP}...')

	fig = plt.figure(dpi = __RES_DPI)
	plt.title("Figure 2A")

	if len(a_rates) == 0 or len(a_errors) == 0:
		a_rates = firing_rates_all_neurons_all_runs[:]
		a_errors = stds_all_neurons_all_runs[:]
	else:
		a_rates = a_rates[:]
		a_errors = a_errors[:]

	# x_axis = np.array([x for x in stims])
	x_axis = np.array([x*10 for x in range(-9,10)])
	y_axis = np.array([x for x in a_rates])
	y_errors = np.array([x/np.sqrt(10) for x in a_errors])

	preference = np.argmax(y_axis)*10

	(nrn_pref, nrn_OSI, nrn_width, _ , _ ) = nf.tuning_properties(y_axis, [x*10 for x in range(0,18)])

	print(f'OSI/width analysis: preferred {nrn_pref} deg, OSI {nrn_OSI}, width {nrn_width}')
	
	stats = lambda data: f'{np.mean(data):.2f}+/-{np.std(data):.2f}'
	print(f'OSI/width statistics: preferred {stats(prefs)} deg, OSI {stats(OSIs)}, width {stats(widths)}, Rpref {stats(rates_pref)} Hz, Rorth {stats(rates_orth)} Hz')

	print(f'Mean preferred orientation for disparity {_DISP}, condition {_CONDITIONS[_CND]}: {preference}')

	# print(np.shape(x_axis), np.shape(y_axis), np.shape(y_errors))

	plt.errorbar(x_axis, y_axis[_TRANSFORM_VECTOR], y_errors[_TRANSFORM_VECTOR], capsize=5, c='k')
	plt.xticks(x_axis)
	plt.xlabel('Stimulus orientation ($^\circ$)')
	plt.ylabel('Neuronal response (Hz)')
	# plt.legend([f'Preference: {preference}'])

	fig.set_tight_layout(True)
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()

	plt.show()

