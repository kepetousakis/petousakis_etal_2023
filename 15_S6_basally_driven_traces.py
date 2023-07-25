import pandas as pd
import numpy as np
import Code_General_utility_spikes_pickling as util
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import warnings
import logging

logging.getLogger('matplotlib.font_manager').disabled = True

warnings.filterwarnings("ignore")

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

mpl.rc('font', **font)

__RES_DPI = 48
_DOTSIZE = 16
_FORMAT = 'eps' 
_SHOW_TEXT_LABELS = True
_TEXT_SHOWS_DIST = False
_BASAL_ORDER_DISTANCE = True
_YLIM = [-80,600]
target_orient = 0
target_neuron = 0
target_run = 0
target_type = 'basal' # 'apical' or 'basal' or 'unstable'
target_intv = 'na' # 'syn' or 'na'
target_lookup = True  # If this is True, then the orient/neuron/run parameters are overriden, and the code looks for a match in the "verdicts" dict
if target_intv == 'na':
	window_shape = (8, 2) # (6,4)  # in milliseconds, before and after somatic spike
elif target_intv == 'syn':
	window_shape = (8, 2)  # in milliseconds, before and after somatic spike
else:
	raise Exception
dt = 0.1
visual_offset = 60 # millivolts

representation = lambda o,n,r: f'Orient{o}_nrn{n}_run{r}'

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


def lookup(target_dict, search_parameter):
	
	keys = target_dict.keys()
	matches = {}
	for key in keys:
		if search_parameter in key:
			matches[key] = target_dict[key]
	return matches


def find_spikes_by_type(verdicts, target_type):

	matching_spikes = []

	for key in verdicts.keys():
		print(key, verdicts[key]['Verdict'])
		if verdicts[key]['Verdict'] == target_type:
			matching_spikes.append(key.split('-'))

	return matching_spikes



paths = find_paths(conn_matrix, dendrite_information)

compartment_names = dendrite_names + ['soma']


try:
	na_verdicts = util.pickle_load('./Data/S6/Data_S6_spike_survival_verdicts_na_intv.pickle')
except:
	print('Data files not found.')
	
if target_intv == 'na':
	verdicts = na_verdicts
else:
	raise Exception

trace_data = np.zeros(shape=(51,25001))

if target_lookup:
	print(f'Looking for targets for type {target_type}...')
	matching_spikes = find_spikes_by_type(verdicts, target_type)
	print(f'Found {len(matching_spikes)} matches. Selecting match #0 ({matching_spikes[0]})...')
	target_neuron = int(matching_spikes[0][1])
	target_run = int(matching_spikes[0][2])
	target_orient = int(matching_spikes[0][3])

if not os.path.exists(f'./Data/S6/{representation(target_orient,target_neuron,target_run)}/soma.dat'):
	raise Exception('Required files not found.')
else:
	for i,name in enumerate(compartment_names):
		trace_data[i] = np.loadtxt(f'./Data/S6/{representation(target_orient,target_neuron,target_run)}/{name}.dat')

#%% ===============================
# Get suitable events
keygen = lambda cnd,nrn,run,stim: f'{cnd}-{nrn}-{run}-{stim}-'
key_partial = keygen(600,target_neuron,target_run,target_orient)
matches = lookup(verdicts, key_partial)

match = (None, None)

for key in matches.keys():
	if not matches[key]['Verdict'] == target_type:
		# del matches[key]
		pass
	else:
		match = (key,matches[key])
		break

# Assuming 1st match is picked (see 'break' statement above)
if match[0] == None:
	print(f'<!> Warning: No spikes in dataset {representation(target_orient,target_neuron,target_run)} match verdict "{target_type}".')
	raise Exception
else:
	spike_idx = match[0][-1]

# Find somatic spikes
(spikes,timings) = util.GetSpikes(trace_data[-1][:], threshold=-20, detection_type='max')
timing = timings[int(spike_idx)]
timing_window = (timing-int(window_shape[0]/dt), timing, timing+int(window_shape[1]/dt))
x_axis = [i*dt for i,x in enumerate(range(timing_window[0], timing_window[2]))]
y1 = timing_window[0]; y2 = timing_window[2]

# Image paths
if _BASAL_ORDER_DISTANCE:
	dist_dict = {x:dendrite_information[f'basal{x}']['distance'] for x in range(0,7)}
	dist_list = [dendrite_information[f'basal{x}']['distance'] for x in range(0,7)]
	sorted_idx = np.argsort(dist_list)  # Returns indices that would result in a sorted dist_list
	true_indices = sorted_idx
else:
	true_indices = range(0,7)

vo = visual_offset

layout = [['.','.','C'],
	      ['.','.','F'],
	      ['.','.','I']]

fig, axes = plt.subplot_mosaic(layout, dpi = __RES_DPI)

# fig = plt.figure(dpi=__RES_DPI)
plt.sca(axes['C'])
plt.title('Supp.Figure 6C')
plt.plot(x_axis, trace_data[-1][y1:y2], c='b')
if _SHOW_TEXT_LABELS:
	plt.text(0, np.mean(trace_data[-1][y1:y2])+5, 'soma')
modpath = mcolors.CSS4_COLORS
colormapping = lambda x: 'r' if np.mod(x,2)==0 else 'k'
colormap_basal = ['black', 'red', 'lime', 'orange', 'saddlebrown', 'magenta', 'yellow']
clrs = [modpath[x] for x in colormap_basal]
colormap_basal = clrs
for enum,i in enumerate(true_indices):
	plt.plot(x_axis, trace_data[i][y1:y2]+vo*(enum+1), c=colormap_basal[i])
	if _SHOW_TEXT_LABELS:
		addon = f' | d={dendrite_information[f"basal{i}"]["distance"]:.01f} um' if _TEXT_SHOWS_DIST else ''
		plt.text(0, np.mean((trace_data[i][y1:y2]+vo*(enum+1)))+5, f'basal{i}{addon}')
	plt.scatter(np.argmax(trace_data[i][y1:y2]+vo*(enum+1))*dt, np.max(trace_data[i][y1:y2]+vo*(enum+1)), c='r', s=_DOTSIZE, zorder=99)
plt.axvline((timing_window[1]-timing_window[0])*dt, ls='--', c='m')
plt.xlabel('Time (ms)')
plt.ylabel('Offset compartment Vm (mV)')
plt.ylim(_YLIM)
fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()



# Issue: many apical paths are very similar to each other.
# Solution: plot paths in a first-come-first-serve manner, then store the plotted paths.
# Every subsequent path is checked for length. If an already plotted path has the same length,
# then we do an elementwise subtraction of the two paths, and check the sum of the resulting list.
# If the sum is over a set number (10), the two paths should be sufficiently different. 
# Otherwise, we skip the similar path.
# The number (10) is approximately the max number of branches in any given subtree of the apical dendrite.
# meaning that no two branches that belong in the same subtree should have indices whose difference
# is greater than 10.

plotted_paths = []

paths_to_image = [[14, 13, 12, 11, 10, 9, 8, 7, 50],[41, 37, 36, 28, 8, 7, 50]]
id2layout = ['F','I']
titles = ['Supp.Figure 6F','Supp.Figure 6I']

def compare_paths(path, path_list, cutoff=10):

	paths_are_similar = False
	found_match = None
	diff = np.inf

	L = len(path)

	if len(path_list) == 0:
		return (paths_are_similar, found_match, diff)
	else:
		for p in path_list:
			if len(p) == L:
				criterion = sum([np.abs(x-y) for x,y in zip(p,path)])
				if criterion <= cutoff:
					paths_are_similar = True
					found_match = p
					diff = criterion
					return (paths_are_similar, found_match, diff)

	return (paths_are_similar, found_match, diff)

idx = -1
for i,path in enumerate(paths):
	# indices 0-6 are basals, 7-49 are apicals, 50 is soma
	# all basals go into the same plot (see prior code)
	if path in paths_to_image:
		idx += 1
		if np.sort(path)[0] < 7:
			print(f'Skipping path {path} (contains basal dendrite)')
		else:
			print(path)
			# Basals could be listed in order without issues, but for apicals we need to invert the path after removing the soma, then list in that order
			if path not in plotted_paths:
				(paths_are_similar, found_match, diff) = compare_paths(path, plotted_paths)
			if not paths_are_similar:

				plotted_paths.append(path)
				path = path[0:-1]  # remove soma (element 50)
				path.reverse()  # reverse list
				path_translated = [x-7 for x in path]
				
				# fig = plt.figure(dpi=__RES_DPI)
				plt.sca(axes[f'{id2layout[idx]}'])
				path_str = '-'.join([str(x-7) for x in path])  # for output file name - "-7" used to convert generic index into apical index
				plt.title(titles[idx])
				# plt.title(f'Apical path {path_translated}')
				plt.plot(x_axis, trace_data[-1][y1:y2], c='b')
				if _SHOW_TEXT_LABELS:
					plt.text(0, trace_data[-1][y1:y2][0]+5, 'soma')
				for i,element in enumerate(path):
					plt.plot(x_axis, trace_data[element][y1:y2]+vo*(i+1), c=colormapping(i))
					if _SHOW_TEXT_LABELS:
						addon = f' | d={dendrite_information[f"apical{element-7}"]["distance"]:.01f} um' if _TEXT_SHOWS_DIST else ''
						plt.text(0, np.mean((trace_data[element][y1:y2]+vo*(i+1)))+5, f'apical{element-7}{addon}')		
					plt.scatter(np.argmax(trace_data[element][y1:y2]+vo*(i+1))*dt, np.max(trace_data[element][y1:y2]+vo*(i+1)), c='r', s=_DOTSIZE, zorder=99)
				plt.axvline((timing_window[1]-timing_window[0])*dt, ls='--', c='m')
				plt.xlabel('Time (ms)')
				plt.ylabel('Offset compartment Vm (mV)')
				plt.ylim(_YLIM)
				fig.set_tight_layout(True)
				figManager = plt.get_current_fig_manager()
				figManager.window.showMaximized()
			else:
				print(f'<!> Skipping path {path}: similar to already existing path {found_match} (diff:{diff})')


plt.show()