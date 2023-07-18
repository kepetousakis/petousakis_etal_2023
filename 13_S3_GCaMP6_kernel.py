# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:38:07 2020

@author: KEPetousakis
"""


from numpy import exp as exp
import numpy as np
from matplotlib import pyplot as plt
from random import randint
import matplotlib as mpl

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

mpl.rc('font', **font)

__RES_DPI = 48


SCALE_COEFF = 370
T_RISE = 0.2 * SCALE_COEFF
T_DECAY_HALF = 0.6 / (SCALE_COEFF*3.75)
T_DECAY = (np.log(2)/T_DECAY_HALF) /2
SCALEFACTOR = 2/3.5
T0 = 0
F_RISE = 1
F_DECAY = 1
DT = 0.025

# -((exp(t-t0)/trise) - (exp(t-t0)/tdecay)  # difference of exponentials kernel, t0 is event timing 

kernel = lambda t, t0, t_rise, t_decay, scalefactor, f_rise, f_decay: -scalefactor*( ( f_rise*exp( -(t-t0)/t_rise) ) - ( f_decay*exp( -(t-t0)/t_decay) ) )
component1 = lambda t, t0, t_rise, scalefactor, f_rise: scalefactor-scalefactor*(f_rise*exp(-(t-t0)/t_rise))
component2 = lambda t, t0, t_decay, scalefactor, f_decay: scalefactor*(f_decay*exp(-(t-t0)/t_decay))

times = [x*DT for x in range(0,int(2500/DT))]

values = [kernel(x,T0,T_RISE, T_DECAY, SCALEFACTOR, F_RISE, F_DECAY) for x in times]
v1 = [component1(x,T0,T_RISE,SCALEFACTOR,F_RISE) for x in times]
v2 = [component2(x,T0,T_DECAY,SCALEFACTOR,F_DECAY) for x in times]


# Kernel component visualization (not required)
# plt.plot(times, values)
# plt.plot(times, v1, '--g')
# plt.plot(times, v2, '--r')
kernel_max = max(values)
kernel_max_pos = np.argmax(values)
kernel_max_plus_06 = values[kernel_max_pos + int(600/DT)]
kernel_decay_06 = (kernel_max_plus_06/kernel_max)*100
print(f'Maximum reached at {np.argmax(values)*DT} ms (t_rise = 0.2 s). Values decayed by {kernel_decay_06}% within 0.6 s (t_decay 1/2 = 0.6 s).')



dummy_data = [randint(0,1) for x in times]

dummy_data = [0 for x in times];

def genspikeburst(totaltime,nspikes,interval,tstart):
	
	starting_trace = totaltime
	new_trace = starting_trace[0:tstart]
	burst = []
	for spike in range(0,nspikes):
		burst.append(1)
		for isi in range(0, interval):
			burst.append(0)
	for element in burst:
		new_trace.append(element)
	for element in starting_trace[tstart+len(burst)-1:]:
		new_trace.append(element)
		
	print(f'Registering burst of {nspikes} spikes starting at {int(tstart*DT)} ms (fixed inter-spike interval of {int(interval*DT)} ms).')
	
	return new_trace

# n_spikes = 10
# ISI = 137 / DT
# spike_timing = 500 / DT
	
delta_F_peaks = [0]
spike_numbers = [1,2,3,4,5]

layout = [['.','A','.'],
	      ['.','B','.']]

fig, axes = plt.subplot_mosaic(layout, dpi = __RES_DPI)
	
for nsp in spike_numbers:
	
	n_spikes = nsp
	ISI = (250/n_spikes) / DT
	spike_timing = 500 / DT
	
	dummy_data = genspikeburst(dummy_data, n_spikes, int(ISI), int(spike_timing))
	
	convolved_data = np.convolve(dummy_data, values)
	
	processed_data = convolved_data[0:len(times)]
	
	#delta_F = [x - processed_data[i-1] if i>0 else 0 for i, x in enumerate(processed_data)]
	delta_F = processed_data
	delta_F_peaks.append(max(delta_F))
	
	# Supplementary Figure 3A
	if nsp == 1:
		# fig = plt.figure(dpi = __RES_DPI)
		plt.sca(axes['A'])
		xax = [x for x in times]
		plt.plot(xax, processed_data)
		plt.axvline(500, c='r')
		plt.title('Supp.Figure 3A')
		plt.xlabel('Time (ms)')
		plt.ylabel('Response intensity (ΔF/F0)')
		fig.set_tight_layout(True)
		figmanager = plt.get_current_fig_manager()
		figmanager.window.showMaximized()



# Supplementary Figure 3B
# fig = plt.figure(dpi = __RES_DPI)
plt.sca(axes['B'])
plt.plot([0]+spike_numbers,delta_F_peaks, marker='o')
plt.title('Supp.Figure 3B')
plt.xlabel('Number of APs in 250 ms'); xticks = spike_numbers
plt.ylabel('Response intensity (peak ΔF/F0)')
fig.set_tight_layout(True)
figmanager = plt.get_current_fig_manager()
figmanager.window.showMaximized()

plt.show()