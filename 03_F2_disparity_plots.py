# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:56:35 2020

@author: KEPetousakis
"""

import os.path 
import numpy as np
import matplotlib.pyplot as plt
import Code_General_utility_spikes_pickling as util
import warnings
from copy import deepcopy as dcp
import Code_General_Nassi_functions as nf
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.colors as mcolors

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

mpl.rc('font', **font)

__RES_DPI = 48

warnings.filterwarnings('ignore')
	
_LOCALPATH = './Data/F2/'

_CNDS = [400,500,600]
_FIGURES = ['3F', '3E', '3D']
_CONDITIONS = {100:"10b:90a",200:"20b:80a",300:"30b:70a",400:"40b:60a",500:"50b:50a", 600:"60b:40a",700:"70b:30a",800:"80b:20a",900:"90b:10a"}
_STIMS = [x*10 for x in range(0,18)]
_DISPS = [x*10 for x in range(0,10)]
_NEURONS = [x for x in range(0,10)]


_BONFERRONI = True

def pref_transform(disp, pref):

	distance_pole = min([abs(pref-disp), abs(pref-disp-180), abs(pref-disp+180)])
	distance_pole_idx = np.argmin([abs(pref-disp), abs(pref-disp-180), abs(pref-disp+180)])
	distance_0 = min([abs(pref-0), abs(pref-0-180), abs(pref-0+180)]) # if e.g. pref = 170, true distance is -10, not 170. If pref is e.g. 20, that is also the true distance.
	distance_0_idx = np.argmin([abs(pref-0), abs(pref-0-180), abs(pref-0+180)]); distance_0_idx += 3 # offset by 2 to correctly find the "distances" index needed
	distances = [(pref), (pref-180), (pref+180), (pref), (pref-180), (pref+180)]
		
	if np.isnan(distance_pole) or np.isnan(distance_0):
		print('Detected untuned neuron')
		return np.nan

	if distance_pole < distance_0:
		return distances[distance_pole_idx]

	if distance_pole > distance_0:
		return distances[distance_0_idx]

	if distance_0 == distance_pole:
		return distances[distance_0_idx]
	
def remove_nans(data, details=''):
	values = []
	indices = []
	
	for i,datum in enumerate(data):
		if not np.isnan(datum):
			values.append(datum)
			indices.append(i)
		else:
			print(f'Removed NaN value from dataset {details}')
			
	return (values, indices)


if __name__ == "__main__":

	try:
		filename_ref  = f'{_LOCALPATH}Data_F3_expectation_lines.pickle'
		
		if os.path.exists(filename_ref):
			references = util.pickle_load(filename_ref)
		else:
			raise Exception
			

		nrn_prefs_all = [ [ [ [] for z in _NEURONS ] for y in _DISPS ] for x in _CNDS ]	
		integrals = [ [ [] for y in _CNDS ] for x in _NEURONS ]	
		diff_int_only = [ [ [] for y in _CNDS ] for x in _NEURONS ]

		distance_fromref_all      = np.empty( shape = ( len(_CNDS), len(_DISPS), len(_NEURONS) ) ) ; distance_fromref_all[:] = np.nan
		distance_fromref_filtered = [ [ [] for y in _NEURONS ] for x in _CNDS]
		distance_fromref_avg_disp = np.empty( shape = ( len(_CNDS), len(_NEURONS) ) ) ; distance_fromref_avg_disp[:] = np.nan
		distance_fromref_std_disp = np.empty( shape = ( len(_CNDS), len(_NEURONS) ) ) ; distance_fromref_std_disp[:] = np.nan
		distance_fromref_avg_disp_nrn = np.empty( shape = ( len(_CNDS) ) ) ; distance_fromref_avg_disp_nrn[:] = np.nan
		distance_fromref_std_disp_nrn = np.empty( shape = ( len(_CNDS) ) ) ; distance_fromref_std_disp_nrn[:] = np.nan

		layout = [['.','D',],
	      		  ['E','F']]
		id2layout = ['F','E','D']

		fig, axes = plt.subplot_mosaic(layout, dpi = __RES_DPI)
		
		for idx_c, cnd in enumerate(_CNDS):
			filename_avgs = f'{_LOCALPATH}/cnd{cnd}_neuron_disparity_rates_filtered.pickle'
			# filename_avgs = f'{_LOCALPATH}/cnd{cnd}_noise{cnd}_neuron_disparity_rates_filtered.pickle'  #test2
				
			if os.path.exists(filename_avgs):
				rates = util.pickle_load(filename_avgs)

			else:
				raise Exception
				
			prefs = []
			stdevs = []
			errors = []
			scatter_points = []
			gauss_prefs = []
			
			for idx_d, disp in enumerate(_DISPS):
				nrn_prefs = []
				for idx_n, nrn in enumerate(rates[idx_d]):
					if len(nrn) > 0:
						idx = np.argmax(nrn)
						nrn_prefs.append(_STIMS[int(idx)])
						nrn_prefs_all[idx_c][idx_d][idx_n] = _STIMS[int(idx)]
					else:
						print(f"Untuned: condition {cnd} neuron {idx_n} disparity {idx_d} responses {nrn} length {len(rates[idx_d])}")
						aleph = dcp(rates[idx_d])
						bet = dcp(rates)
						nrn_prefs.append(np.nan)
						
				for i,pref in enumerate(nrn_prefs):
					fixed_pref = pref_transform(disp, pref)
					nrn_prefs[i] = fixed_pref
					nrn_prefs_all[idx_c][idx_d][i] = fixed_pref					
					
					
				nrn_prefs = remove_nans(nrn_prefs, details = f'{cnd}|{disp}')[0]	
				scatter_points.append(nrn_prefs)
				mean_pref = np.mean(nrn_prefs)
				# Temp change
				rates_filt = []
				for e in rates[idx_d]:
					if not len(e) == 0:
						rates_filt.append(e)
				if not len(rates_filt) == 0:
					averaged_rates = np.mean(rates_filt, axis=0) # dimensions are disp, nrn, stim - disp is selected, average across nrn (idx 0)
					(avg_pref, avg_OSI, avg_width, _ , _ ) = nf.tuning_properties(averaged_rates, [x*10 for x in range(0,18)])
					print(f'<!> Gaussian fit results for disp {disp}: {avg_pref}')
					gauss_prefs.append(avg_pref)
				else:
					print(f'<!> All neurons untuned in disp {disp}')

				std_pref = np.std(nrn_prefs)
				err_pref = np.std(nrn_prefs)/np.abs(np.sqrt(len(nrn_prefs)))
				prefs.append(mean_pref)
				stdevs.append(std_pref)
				errors.append(err_pref)
				
			
			xax = [x for x in range(0,100,10)]
			xax2 = [x for x in range(0,100,10)]
			
			yax = [x for x in prefs if not np.isnan(x)]
			errs = [x for x in errors if not np.isnan(x)]
			if len(yax) < len(xax2):
				xax2 = xax[0:len(yax)]
			
			# Figures 3D, 3E and 3F

			# fig = plt.figure(dpi=__RES_DPI)
			plt.sca(axes[f'{id2layout[idx_c]}'])
			plt.title(f'Figure {_FIGURES[idx_c]} ({_CONDITIONS[_CNDS[idx_c]]})')
			if len(_CNDS) < 9:   # for plotting single curves
				idx_c_ref = int(cnd/100)-1
			else:
				idx_c_ref = idx_c
			plt.plot(xax[0:], references[0,idx_c_ref,0:], '--k', linewidth=2)
			plt.errorbar(xax2[0:len(yax)], yax, errs, fmt='.-b', linewidth=2)
			# plt.plot(xax2[0:len(gauss_prefs)], gauss_prefs, c='r', ls='dotted', zorder=0)
			
			for i,points in enumerate(scatter_points):
				xax3 = [i*10 for x in range(0,len(points))]
				for value in set(points):
					counter = 0
					for j,ypos in enumerate(points):
						if ypos == value and counter > 0:
							xax3[j] = xax3[j] + (0.5*counter)
							counter +=1
						elif ypos == value:
							counter +=1
				
			alldistances_fortext = []	
			for d,n_p in enumerate(scatter_points):  # scatter_points: 10 disparities, 10 neurons inside each one
				distances = [references[0,idx_c_ref,d] - x for x in n_p]
				distance = np.mean(distances)
				alldistances_fortext.append(distance)
				
			plt.axhline(0,0,1,c='k',ls='--')
			plt.xlabel('Disparity ($^\circ$)', fontsize=18)
			plt.ylabel('Neuronal orientation preference ($^\circ$)', fontsize=18)
			plt.xticks(xax, fontsize=18)
			plt.yticks(xax, fontsize=18)
			plt.xlim(0,10*(len(yax)-1)+5)
			plt.ylim(-5,max(max(yax)+5,40))
			plt.fill_between(xax[0:len(yax)], yax, references[0,idx_c_ref,0:len(yax)], alpha=0.2, facecolor='k')
			fig.set_tight_layout(True)
			figmanager = plt.get_current_fig_manager()
			figmanager.window.showMaximized()
			
			for idx_n in _NEURONS:
				indices1 = [x for x in range(0,len(yax))]
				intgr_ref = np.trapz(references[0,idx_c_ref,:][indices1], np.array(xax)[indices1])
				intgr_act = np.trapz(np.array(yax)[indices1], np.array(xax)[indices1])
				
				for idx_d, disp in enumerate(_DISPS):
					try:
						distance_fromref_all[idx_c, idx_d, idx_n] = references[0,idx_c_ref,idx_d] - nrn_prefs_all[idx_c][idx_d][idx_n]
					except:
						distance_fromref_all[idx_c, idx_d, idx_n] = np.nan

				diff_integral = intgr_ref - intgr_act
				if diff_integral < 0:
					label = 'basal bias'
				elif diff_integral > 0:
					label = 'apical bias'
				else:
					label = 'no bias'
				
				integrals[idx_n][idx_c] = (intgr_ref,intgr_act,diff_integral)
				diff_int_only[idx_n][idx_c] = diff_integral

	except Exception as e:
		print(e)
		raise Exception
		
	else:
		print('No errors during script execution.')
		
	finally:
		plt.show()

