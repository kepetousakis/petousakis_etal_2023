import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import Code_General_utility_spikes_pickling as util
from copy import deepcopy as dcp
import scipy.ndimage as scp
from numpy import exp as exp
import matplotlib as mpl

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

mpl.rc('font', **font)

__RES_DPI = 48

# cnd = 600
# nrn = 2
# run = 0
cnd = 600; nrn = 1; run = 2;
target = 'soma'

SCALE_COEFF = 370
T_RISE = 0.2 * SCALE_COEFF
T_DECAY_HALF = 0.6 / (SCALE_COEFF*3.75)
T_DECAY = (np.log(2)/T_DECAY_HALF) /2
SCALEFACTOR = 2/3.5
T0 = 0
F_RISE = 1
F_DECAY = 1
DT = 0.1
DEBUG = True

_DETECTION_VALUE_FACTOR = 0.935
_DETECTION_DERIVATIVE_FACTOR = 0.67
_BINARIZATION_NOISE = False
_BINARIZATION_NOISE_RANGE = 0.05

frame_time = 0.138716725543428

def binarize(trace, random_noise = _BINARIZATION_NOISE):
	spikes, timings = util.GetSpikes(trace,0, False, 'max')
	if not random_noise:
		event_raster = [1 if x in timings else 0 for x in range(0,len(trace))]
	else:
		nr = _BINARIZATION_NOISE_RANGE
		event_raster = [1+rnd.uniform(0,nr) if x in timings else rnd.uniform(0,nr) for x in range(0,len(trace))]
	for x,y in enumerate(event_raster):
		if y == 1:
			print(f'Spike at timepoint {x}')
	return event_raster


def downsample(vector, origin_dt = 0.1/1000, target_dt = 0.138):

	downsampling_indices = [x*(target_dt/origin_dt) for x in range(0,int(len(vector)/(target_dt/origin_dt)+1))]

	downsampled_vector = [vector[int(x)] for x in downsampling_indices]

	return downsampled_vector


def gcamp6s_kernel(t, t0 = T0, t_rise = T_RISE, t_decay = T_DECAY, scalefactor = SCALEFACTOR, f_rise = F_RISE, f_decay = F_DECAY):

	return -scalefactor*( ( f_rise*exp( -(t-t0)/t_rise) ) - ( f_decay*exp( -(t-t0)/t_decay) ) )


def transformtrace(Fvalues, strictness = 'dynamic'):
	"""Centers a fluorescence trace by subtracting the mean, then returns that centered trace alongside a weighted standard deviation."""

	mean_val = sum(Fvalues)/len(Fvalues)

	stdev = np.std(Fvalues)

	Tvalues = Fvalues-mean_val

	if strictness == 'dynamic':
		stdev_multiplier = 3
		lowest_percentage = 0.1  

		std_values = dcp(Fvalues)
		std_values.sort()
		nvalues = int(len(Fvalues)*lowest_percentage)
		if nvalues < 2:
			nvalues = 2
		std_values = std_values[0:nvalues]
		stdev = np.std(std_values)
		return(Tvalues, stdev*stdev_multiplier)

	if strictness == "window":

		frame_time = 0.138716725543428
		stdev_multiplier = 3
		window_size_seconds = 2
		window_size = int(window_size_seconds/frame_time)  # seconds expressed as frames, rounded to the nearest frame
		# Exact implementation: we use 2 windows, one to the left of the reference frame, and one to the right. We also include the reference frame.
		# We take the std of all windows + the reference frame, and use that x3 as the threshold, for that particular frame.
		thresholds = []

		for i,frame in enumerate(Fvalues):
			left_bound = i - window_size

			if left_bound < 0:
				left_bound = 0

			window_left = Fvalues[left_bound:i]  # Defines interval [left_bound, i)
			window_left = [x for x in window_left]
			right_bound = i+window_size

			if right_bound > len(Fvalues):
				right_bound = len(Fvalues)-1

			window_right = Fvalues[i:right_bound+1] # Defines interval [i, right_bound], so it includes the reference frame (i)
			window_right = [x for x in window_right]

			window = [x for x in window_left]
			for x in window_right:
				window.append(x)

			frame_std = np.std(window)
			thresholds.append(frame_std*stdev_multiplier)

		return(Tvalues, thresholds)


	return(Tvalues, stdev*stdev_multiplier)


def derivative_getspikes(data, value_threshold='min_percent', derivative_threshold='mean_strict'):

	if value_threshold == 'mean':
		value_threshold = np.mean(data)
	elif value_threshold == 'nonzero_mean':
		m = np.mean([x for x in data if x > 0])
		temp = [x for x in data if x > m]
		value_threshold = min(temp)
	elif value_threshold == 'std':
		value_threshold = np.std(data)
	elif value_threshold == 'min_percent':
		percent = _DETECTION_VALUE_FACTOR
		temp = np.sort(data)
		# temp = [x for x in temp if x>0]
		value_threshold = np.max(temp[0:int(len(temp)*percent)])
		if value_threshold == data[0]:
			return (0, [], value_threshold, derivative_threshold)

	data_deriv = np.diff(data, prepend = data[0])

	if derivative_threshold == 'mean':
		derivative_threshold = np.mean(data_deriv)
	elif derivative_threshold == 'mean_strict':
		temp = np.sort(data_deriv)
		percent = _DETECTION_DERIVATIVE_FACTOR
		derivative_threshold = np.max(temp[0:int(len(temp)*percent)])

	if value_threshold == data[0] and derivative_threshold == data_deriv[0]:  # if no spikes at all, this avoids detecting an artificial spike
		return (0, [], value_threshold, derivative_threshold)

	nspikes = 0
	timings = []
	sidx = []
	eidx = []
	in_spike = False

	for x, (value, derivative) in enumerate(zip(data, data_deriv)):
		if value >= value_threshold and derivative >= derivative_threshold and not in_spike:
			nspikes += 1
			timings.append(x)
			sidx.append(x)
			in_spike = True
		if in_spike:
			if value < value_threshold:
				eidx.append(x)
				in_spike = False
			if derivative < derivative_threshold:
				eidx.append(x)
				in_spike = False
		if in_spike and x == len(data)-1:
			eidx.append(x)
			in_spike = False

	return (nspikes, timings, value_threshold, derivative_threshold)

kernel = [gcamp6s_kernel(x*DT) for x in range(0,int(5/(DT/1000))+1)]

try:
	data = np.loadtxt('./Data/S2/Data_F2B_soma_trace.dat')
except:
	raise Exception

xticks = [x*DT for x in range(0,len(data))]

print(len(xticks),len(data))

# Figure 2B
fig = plt.figure(dpi = __RES_DPI)
plt.suptitle('Figure 2B')
plt.subplot(311)
plt.plot(xticks,data)
plt.ylabel('Vm (mV)')

plt.subplot(312)
data = binarize(data)
for x,element in enumerate(data):
	if element == 1:
		plt.axvline(x=x*DT, color='r')
		
redline_data = dcp(data)


data = np.convolve(data, kernel)
#data = data[0:100001]
data = data[0:int(5/(DT/1000))+1]

plt.plot(xticks,data)
plt.ylabel('Raw fluorescence\n approx. (ΔF/F0)')

data = downsample(data, origin_dt=DT/1000)
xticks = [x*0.138716725543428*1000 for x in range(0,len(data))]
plt.subplot(313)
data,threshold = transformtrace(data)
plt.plot(xticks,data)
plt.ylabel('Downsampled\n approx. (ΔF/F0)')
plt.xlabel('Time (ms)')
for x,element in enumerate(redline_data):
	if element == 1:
		plt.axvline(x=x*DT, color='r')

fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()