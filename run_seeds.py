import numpy as np
import random
import math
import sys
import os
import pandas
import numpy as np
import scipy
import matplotlib.pyplot as plt
import argparse

import lib_local as lib

def write_input_from_crds(crd_s, epsilon = lambda i: 10.0, filepath=None):
	n = crd_s.shape[0]
	# lines = ['ID type Epsilon x y z'] + \
		# ['%2d %2d %15lf %15lf %15lf %15lf' % (i+1, i+3, epsilon(i), crd_s[i, 0], crd_s[i, 1], crd_s[i, 2]) for i in range(n)]
	lines = ['ID type Epsilon x y z'] + \
		['%d %d %lf %lf %lf %lf' % (i+1, i+3, epsilon(i), crd_s[i, 0], crd_s[i, 1], crd_s[i, 2]) for i in range(n)]
	
	if(filepath is not None):
		open(filepath, 'w').write('\n'.join(lines))
	
	return lines

def n_from_angle(phi, tht):
	return np.array([np.sin(tht) * np.cos(phi), np.sin(tht) * np.sin(phi), np.cos(tht)])

def write_input_from_angles(phi_s, tht_s, epsilon = lambda i: 10.0, R=4.0, filepath=None):
	n = len(tht_s)
	assert(n == len(phi_s)), 'ERROR: len(phi_s) must be == len(tht_s)'
	
	xyz_s = np.array([n_from_angle(phi_s[i], tht_s[i]) for i in range(n)]) * R
	# [np.sin(tht_s[i]) * np.cos(phi_s[i]), np.sin(tht_s[i]) * np.sin(phi_s[i]), np.cos(tht_s[i])]
	
	return write_input_from_crds(xyz_s, epsilon=epsilon, filepath=filepath)

def generate_normals(tht_fnc, n=24, site_sgm=1.0/4, i_max=-200, to_plot=False):
	dim = 3
	#phi = np.random.random(n) * (2 * np.pi)
	
	tht_fnc_renorm = lambda t: tht_fnc(t) * np.sin(t)
	tht_fnc_z = scipy.integrate.quad(tht_fnc_renorm, 0, np.pi)[0]
	
	tht_opt = scipy.optimize.minimize_scalar(lambda t: -tht_fnc_renorm(t), bounds=(0, np.pi))
	tht_fnc_opt_t = tht_opt.x
	tht_fnc_opt_max = -tht_opt.fun
	
	assert(tht_fnc_opt_max > 0), 'ERROR: fnc_max <= 0'
	
	#tht_fnc_renorm = lambda t: tht_fnc(t) * np.sin(t) * (tht_fnc_opt_max / tht_fnc_z)
	tht_fnc_renorm = lambda t: tht_fnc(t) * np.sin(t) / tht_fnc_opt_max
	
	normals = np.empty((n, dim))
	sgm2 = site_sgm**2
	i = 0
	if(i_max < 0):
		i_max = -i_max * n
	
	if(to_plot):
		t_draw = np.linspace(0, np.pi, 1000)
		
		fig, ax, _ = lib.get_fig(r'$\theta$', r'$\sim p_{\theta}$')
		ax.plot(t_draw, tht_fnc_renorm(t_draw))
		lib.add_legend(fig, ax)
		
		fig2, ax2, _ = lib.get_fig(r'$\theta$', r'$\sim p_{\theta} / \sin(\theta)$', title=r'$p_{\theta, norm}$')
		ax2.plot(t_draw[1:-1], tht_fnc_renorm(t_draw[1:-1]) / np.sin(t_draw[1:-1]))
		lib.add_legend(fig2, ax2)
		
		plt.show()
	
	i_total = 0
	while(i < n):
		tht_new = lib.np_rng.random() * np.pi
		if(lib.np_rng.random() < tht_fnc_renorm(tht_new)):
			phi_new = lib.np_rng.random() * 2 * np.pi
			normal_new = n_from_angle(phi_new, tht_new)
			
			no_overlaps = (site_sgm == 0) or (i == 0)
			if(not no_overlaps):
				no_overlaps = np.all(np.sum((normals[ : i, :] - np.broadcast_to(normal_new[None, :], (i, dim)))**2, axis=1) > sgm2)
			if(no_overlaps):
				normals[i, :] = normal_new
				i += 1
		
		i_total += 1
		if(i_total >= i_max):
			break
	
	assert(i == n), 'ERROR: unable to fit %d (i_max = %d) normals with sgm=%s into a sphere with given rho' % (n, i_max, lib.f2s(site_sgm))
	
	return normals

def generate_nanoparticle(filepath, tht_fnc=lambda t: 1.0, n=24, R=4.0, site_sgm=1.0, i_max=-200, epsilon=10.0):
	
	normals = generate_normals(tht_fnc, n=n, site_sgm=site_sgm/R, i_max=i_max)
	
	input_lines = write_input_from_crds(normals * R, epsilon = lambda i: epsilon, filepath=filepath)
	
	return input_lines

def main():
	# python run_seeds.py --seed 1 2 3 --R 4.0 --site_sgm 0.1 --epsilon 0.25 --timesteps_total 100000 --n_sites 1000 --orient_mode 3
	
	# python run_seeds.py --seed 1 2 3 --R 4.0 --site_sgm 0.1 --epsilon 0.25 --timesteps_total 100000 --n_sites 1000 --orient_mode 1 --e_mode 2
	
	# ======================= argparse ===================
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--seeds', type=int, nargs='+', help='random seeds')
	parser.add_argument('--timesteps_total', type=int, default=100000, help='prod timesteps')
	parser.add_argument('--orient_mode', type=int, default=1, help='prod timesteps')
	parser.add_argument('--e_mode', type=int, default=1, help='site energy distribution')
	parser.add_argument('--to_gen_normals', type=int, default=1, help='generate input or copy the existing one')
	
	parser.add_argument('--R', type=float, default=4.0, help='R_big')
	parser.add_argument('--site_sgm', type=float, default=0.1, help='site overlaps')
	parser.add_argument('--i_max', type=int, default=-200, help='i_max for site generation')
	parser.add_argument('--epsilon', type=float, default=0.25, help='site-membrane binding')
	parser.add_argument('--n_sites', type=int, default=1000, help='N of sites')
	
	clargs = parser.parse_args()
	
	seeds = np.array(clargs.seeds, dtype=int)
	timesteps_total = clargs.timesteps_total
	orient_mode = clargs.orient_mode
	e_mode = clargs.e_mode
	n_sites = clargs.n_sites
	R = clargs.R
	site_sgm = clargs.site_sgm
	i_max = clargs.i_max
	epsilon = clargs.epsilon
	
	for s in seeds:
		cmd = 'python MakeLammpsInput.py --seed %s --R %s --site_sgm %s --epsilon %s --n_sites %d --orient_mode %d --e_mode %d --timesteps_total %d --to_run_lmp 1 --no_verbose 1' % \
					(s, lib.f2s(R), lib.f2s(site_sgm), lib.f2s(epsilon), n_sites, orient_mode, e_mode, timesteps_total)
		#print(cmd)
		lib.run_it(cmd)

if __name__ == "__main__":
	main()
