# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:29:02 2020

@author: Kepetousakis
"""

import numpy as np
import pickle
from copy import deepcopy as dcp
from pathlib import Path as fpath
from scipy.stats import norm as norm

def Find(condition):
	try:
		return [int(x) for x in np.where(condition)[0]]
	except:
		return 0


def GetSpikes(vtrace, threshold = 0, debug = False, detection_type='onset'):
	"""Given a trace, find the number and timing of spikes therein using a threshold value.
	detection_type: {'onset', 'max', 'offset'}"""

	thrA = threshold  # Detection threshold

	nspikes = 0
	times = []

	# Get suprathreshold values
	sthr = Find(vtrace > thrA)

	if debug:
		print('Suprathreshold before elimination: ', sthr)

	if len(sthr) == 0:  # If no suprathreshold values exist, return nothing
		return(0,[])

	# Suprathreshold values could be successive (i.e. indices 1,2,3,4,5...), but these still represent just one spike. When such a case arises, we need to only keep the highest index.
	successive = 0
	values = dcp(sthr)

	for i,element in enumerate(sthr):  # Iterate through all suprathreshold values
		if not i+1 > len(sthr)-1:  # If we haven't exceeded the max index...
			if debug:
				print(sthr[i+1], element+1, sthr[i+1] == element+1, successive)
			if sthr[i+1] == element+1 and not successive:  # If we find successive elements for the first time (or after removing previously found successive elements)...
				successive = 1  # ...we're now finding successive elements...
				values.remove(element+1) # ... and we remove this unnecessary element from the list of suprathreshold values
			elif sthr[i+1] == element+1 and successive:  # If we've already been finding successive elements, and find another...
				values.remove(element+1) # ... remove this unnecessary element from the list of suprathreshold values
			elif sthr[i+1] != element+1 and successive:  # If the next element is not successive, whereas previous ones were...
				successive = 0  # ...we're no longer finding successive elements, and we don't remove any more for this iteration
			elif sthr[i+1] != element+1 and not successive:
				pass

	sthr = dcp(values)
	if debug:
		print('Suprathreshold after elimination:  ',sthr)

	# Since we get all suprathreshold values (excluding the cases caught right above), then the remainder is subthreshold. So by incrementing/decrementing indices of suprathreshold values,
	# we get the ending and starting point of a spike, respectively. Then, by getting the maximum within that frame, we get the timing of the spike.
	# The problems are:
	# 1) Zero start index
	# 2) len(vtrace) end index
	# 3) Not all suprathreshold values in the sthr variable
	# We'll tackle these in reverse order.

	sidx = [x-1 for x in sthr]  # Start-of-spike indices
	eidx = [x+1 for x in sthr]  # End-of-spike indices

	# Check for suprathreshold values in these index lists
	cleared_start = 0
	cleared_end = 0
	si = 0
	ei = 0

	# Catch negative indices in sidx and over array size indices in eidx
	sidx = [x if x > 0 else 0 for x in sidx]
	eidx = [x if x <= len(vtrace)-1 else len(vtrace)-1 for x in eidx]

	while not cleared_start:
		while vtrace[sidx[si]] > thrA:   # While the "si"-eth value of the list of start indices is suprathreshold...
			sidx[si] -= 1    	 # ...this was not the actual starting index, so we move it back a place.
			if sidx[si] <= 0: 	 # If we've reached a negative index...
				sidx[si] = 0  	 # ...set the starting index to be zero.
				break
		if si == len(sidx)-1: # If we've iterated through all possible starting indices...
			cleared_start = 1 # ...we've "cleared_start" and can move to end indices.
			break
		si += 1

	if debug:
		print('Spike starting indices: ', sidx)

	while not cleared_end:
		while vtrace[eidx[ei]] > thrA:
			eidx[ei] += 1
			if eidx[ei] >= len(vtrace)-1:
				eidx[ei] = len(vtrace)-1
				break
		if ei == len(eidx)-1:
			cleared_end = 1
			break
		ei += 1

	if debug:
		print('Spike ending indices:   ',eidx)

	# We've circumvented the issue of uncaught suprathreshold values by removing the appropriate indices. Now, we have to actually find the exact timing of the spikes.
	spikes = [(x,y) for x,y in zip(sidx,eidx)]
	for spike in spikes:
		start_actual = spike[0]
		end_actual = spike[1]
		while vtrace[start_actual] > thrA:
			start_actual-=1
			if start_actual <= 0:  # Catch cases where the starting index is less than 0
				start_actual = 0
				break
			if vtrace[start_actual] < thrA:
				break

		while vtrace[end_actual] > thrA:
			end_actual+=1
			if end_actual >= len(vtrace)-1:  # Catch cases where the ending index is greater than the length of the trace
				end_actual = len(vtrace)
				break
			if vtrace[end_actual] < thrA:
				break

		if detection_type == 'max':
			vmax = max(vtrace[start_actual:end_actual])
			time = Find(vtrace[start_actual:end_actual]==vmax)[0]
			time = time + start_actual
			times.append(time)
			nspikes += 1
		elif detection_type == 'onset':
			times.append(start_actual+1)
			nspikes+=1
		elif detection_type == 'offset':
			times.append(end_actual)
			nspikes+=1
		else:  # default to max detection
			vmax = max(vtrace[start_actual:end_actual])
			time = Find(vtrace[start_actual:end_actual]==vmax)[0]
			time = time + start_actual
			times.append(time)
			nspikes += 1

	if debug:
		print('All spikes unique? ',len(times)==len(set(times)))  # Casting a list into a set removes duplicate elements and makes it unordered, so we compare their lengths.
		print(f'Found {nspikes} spikes: {times}')		
		
	return (nspikes, times)


def pickle_dump(data, filename='data.pickle'):

	try:
		#if not './' in filename:
		#	filename = './'+ filename

		with open(filename,'wb+') as f:
			pickle.dump(data, f)

	except Exception as e:
		print(f'Pickle failed to dump data.\n{e}')
		return 0

	return 1


def pickle_load(filename):
	
	filename = fpath(filename)

	try:
		with open(filename,'rb') as f:
			return pickle.load(f)
	except Exception as e:
		print(f'Pickle load failed.\n{e}')


def stats_pearson_fisher(n_a, r_a, n_b, r_b):

	z_a = np.arctanh(r_a)
	z_b = np.arctanh(r_b)
	z_obs = (z_a - z_b)/np.sqrt(1/(n_a-3)+1/(n_b-3))
	pval = 2 * norm.cdf(-np.abs(z_obs))
	significant = "*" if pval <= 0.05 else "NS"
	print(f"Comparison of Pearson r-values (r_a: {r_a:.05f} | r_b: {r_b:.05f}) using Fisher's z-transformation | p-value: {pval:.05f}  {significant}")
	return pval