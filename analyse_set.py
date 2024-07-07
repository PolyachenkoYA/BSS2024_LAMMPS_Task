import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

import lib_local as lib # matplotlib has to go before ovito

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
import glob
import sys
import ovito.io
from ovito.io import import_file
import freud


def get_wrapping(p, f):
	d = p.compute(f)
	positions = np.array(d.particles['Position'])
	types = np.array(d.particles['Particle Type'])
	box = freud.Box.from_matrix(d.cell[...])
	np_pos = positions[types == 2]
	mem_pos = positions[types == 1]
	aabb = freud.locality.AABBQuery(box, mem_pos)
	neighbors = aabb.query(np_pos, {'r_max': 5.5})
	# neighbors = [len(n) for n in neighbors]
	neighbors = sum(1 for _ in neighbors)
	# area= np.pi*0.5**2*len(neighbors)
	area= np.pi*0.5**2*neighbors
	spharea = 4*np.pi*4.5**2
	wrapping = area/spharea
	return wrapping

def extract(path, rmax = 1.5):
	# Load trajectory
	p = import_file('%s/output.xyz'%(path))
	# Get number of frames
	nf = p.source.num_frames

	# Initialize cluster object
	cl = freud.cluster.Cluster()
	# Initialize cluster properties object
	cl_props = freud.cluster.ClusterProperties()

	# Initialize list to store number of clusters per frame and frame numbers
	ncl = []
	frs = []
	
	# Loop over frames
	for f in range(nf):
		# Extract data from pipeline frame and keep only membrane positions
		d = p.compute(f)
		pos = np.array(d.particles['Position'])
		typ = np.array(d.particles['Particle Type'])
		mem = pos[typ == 1]
		# Define the box in Freud from the dataframe
		box = freud.Box.from_matrix(d.cell[...])
		# Compute clusters and cluster properties using Freud
		cl.compute((box, mem), neighbors={'r_max': rmax})
		cl_props.compute((box, mem), cl.cluster_idx)
		# Extract cluster sizes
		sizes = cl_props.sizes
		sizes = sizes[sizes > 10]
		# Store number of clusters and frame number
		ncl.append(len(sizes))
		frs.append(f)

	final_wrapping = get_wrapping(p, nf)
	
	# Define pandas dataframe with number of clusters per frame and frame numbers
	data = pd.DataFrame({'frame':frs, 'ncl':ncl})

	# Save dataframe to file and return it
	data.to_csv('%s/clusters.txt'%(path))
	return data, final_wrapping

def analyse_seeds(path, rmax = 1.5):
	# Change directory to path
	r = os.chdir(path)
	# Get list of seeds
	seeds = glob.glob('sd*')
	# Initialize list to store budding frames
	buds = []
	wraps = []
	# Loop over seeds
	for i, seed in enumerate(seeds):
		# Extract clusters from seed
		data, final_wrapping = extract(seed, rmax)
		wraps.append(final_wrapping)
		# Check if budding occurs
		if 2 in data['ncl'].values:
			# If budding occurs, store frame number
			buds.append(data[data['ncl'] == 2].iloc[0]['frame'])
		else:
			# If budding does not occur, store NaN
			buds.append(np.nan)
		
		print(i+1, 'out of', len(seeds), 'done')
	# Build pandas dataframe with budding frames
	buds = pd.DataFrame({'seed':seeds,'frame':buds})
	# Save dataframe to file and return it
	buds.to_csv('budding_frames.txt')
	# Build pandas dataframe with wrapping
	wraps = pd.DataFrame({'seed':seeds,'wrapping':wraps})
	# Save dataframe to file and return it
	wraps.to_csv('wrapping.txt')
	return buds, wraps

if __name__ == '__main__':
	
	# python analyse_set.py --orient_mode 1
	
	# python analyse_set.py --R 4.0 --site_sgm 0.1 --epsilon 0.25 --n_sites 1000 --timesteps_total 100000 --orient_mode 1
	
	# python analyse_set.py --R 4.0 --site_sgm 0.1 --epsilon 0.25 --n_sites 1000 --timesteps_total 100000 --orient_mode 1 -e_mode 2
	
	# ======================= argparse ===================
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--timesteps_total', type=int, default=100000, help='prod timesteps')
	parser.add_argument('--orient_mode', type=int, default=1, help='prod timesteps')
	parser.add_argument('--e_mode', type=int, default=1, help='site energy distribution')
	parser.add_argument('--to_gen_normals', type=int, default=1, help='generate input or copy the existing one')
	
	parser.add_argument('--R', type=float, default=4.0, help='R_big')
	parser.add_argument('--site_sgm', type=float, default=0.1, help='site overlaps')
	parser.add_argument('--i_max', type=int, default=-200, help='i_max for site generation')
	parser.add_argument('--epsilon', type=float, default=0.25, help='site-membrane binding')
	parser.add_argument('--n_sites', type=int, default=1000, help='N of sites')
	
	parser.add_argument('--to_plot_density', type=int, default=1, help='whether to plot sorrespond density')
	parser.add_argument('--dt_dump', type=float, default=100 * 0.01, help='dump time [LJ units]')
	
	clargs = parser.parse_args()
	
	timesteps_total = clargs.timesteps_total
	orient_mode = clargs.orient_mode
	e_mode = clargs.e_mode
	n_sites = clargs.n_sites
	R = clargs.R
	site_sgm = clargs.site_sgm
	i_max = clargs.i_max
	epsilon = clargs.epsilon
	dt = clargs.dt_dump

	model_group_name = lib.model_name_fnc(R, site_sgm, epsilon, n_sites, timesteps_total, orient_mode, e_mode)
	model_group_path = os.path.join(lib.root_path, model_group_name)
	# Define path to parameter set (set of simulations for the same parameters but different RNG seeds)
	#gpath = sys.argv[1]

	# Extract clusters from trajectory
	# data = extract(path)
	data_bud, data_wrap = analyse_seeds(model_group_path)
	
	if(clargs.to_plot_density):
		t_draw = np.linspace(0, np.pi, 1000)
		
		fig, ax, _ = lib.get_fig(r'$\theta$', r'$\sim p_{\theta}$', title=r'$\sim p_{\theta}$; $ID_{\rho}$ = %d' % (orient_mode))
		
		ax.plot(t_draw, lib.tht_fnc_dict[orient_mode](t_draw), label=r'$\sim p_{\theta}$')
		ax.plot([0, np.pi], [0] * 2, '--', label='y=0')
		
		figE, axE, _ = lib.get_fig(r'$\theta$', r'$\sim e_{\theta}$', title=r'$\sim e_{\theta}$; $ID_{e}$ = %d' % (e_mode))
		
		axE.plot(t_draw, lib.tht_e_fnc_dict[e_mode](t_draw), label=r'$\sim p_{\theta}$')
		#axE.plot([0, np.pi], [0] * 2, '--', label='y=0')
		
		lib.add_legend(fig, ax)
		lib.add_legend(figE, axE)
		
	
	# Print frame where budding occurs
	
	print()
	print()
	print(data_bud)
	print('<budding_time / t_LJ> = %s' % lib.errorbar_str(data_bud.loc[:, 'frame'].mean() * dt, data_bud.loc[:, 'frame'].std() / np.sqrt(len(data_bud) - 1) * dt))
	print()
	print()
	print(data_wrap)
	print('<wrapping> = %s' % lib.errorbar_str(data_wrap.loc[:, 'wrapping'].mean(), data_wrap.loc[:, 'wrapping'].std() / np.sqrt(len(data_wrap) - 1)))
	print()
	print()
	
	plt.show()
