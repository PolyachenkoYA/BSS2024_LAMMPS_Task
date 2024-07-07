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

#import mylib as my
import lib_local as lib

#env OMP_NUM_THREADS=2 lmp_della_2022_3_CV_dipole_ser_my -in in.local -sf intel -sf omp
#### OMP and INTEL do not work on both 2016 and 2021
#env OMP_NUM_THREADS=2 lmp_BSS24_1 -in in.local -sf intel -sf omp
#env OMP_NUM_THREADS=2 lmp_BSS24_1 -in in.local -sf omp

# lmp_della_2022_3_CV_dipole_ser_my -in in.local
# lmp_della_2022_3_CV_dipole_ser_my -in in.local &> /dev/null &

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
	tht_s = np.empty(n)
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
				tht_s[i] = tht_new
				i += 1
		
		i_total += 1
		if(i_total >= i_max):
			break
	
	assert(i == n), 'ERROR: unable to fit %d (i_max = %d) normals with sgm=%s into a sphere with given rho' % (n, i_max, lib.f2s(site_sgm))
	
	return normals, tht_s

def generate_nanoparticle(filepath, tht_fnc=lambda t: 1.0, n=24, R=4.0, site_sgm=1.0, i_max=-200, epsilon=10.0):
	
	normals, tht_s = generate_normals(tht_fnc, n=n, site_sgm=site_sgm/R, i_max=i_max)
	
	if(isinstance(epsilon, float)):
		epsilons = lambda i: epsilon
	else:
		eps_arr = epsilon(tht_s)
		#e_norm = (24 * 10.0) / np.sum(eps_arr)
		e_norm = (0.25 * 1000) / np.sum(eps_arr)
		epsilons = lambda i, arr=eps_arr * e_norm: arr[i]
	
	input_lines = write_input_from_crds(normals * R, epsilon = epsilons, filepath=filepath)
	
	return input_lines

def main():
	# python MakeLammpsInput.py --seed 1
	
	# python MakeLammpsInput.py --seed 1000 --R 4.0 --site_sgm 1.0 --epsilon 10.0 --n_sites 24 --orient_mode 1
	
	# python MakeLammpsInput.py --seed 1 --R 4.0 --site_sgm 0.5 --epsilon 10.0 --n_sites 50 --orient_mode 1 --to_run_lmp 0
	# python MakeLammpsInput.py --seed 1 --R 4.0 --site_sgm 0.1 --epsilon 0.25 --n_sites 1000 --orient_mode 4 --to_run_lmp 1
	
	# python MakeLammpsInput.py --seed 1 --R 4.0 --site_sgm 0.1 --epsilon 0.25 --n_sites 1000 --orient_mode 1 --e_mode 1 --to_run_lmp 1 --no_verbose 0
	
	# ======================= argparse ===================
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--seed', type=int, required=True, help='random seed')
	parser.add_argument('--timesteps_total', type=int, default=200000, help='prod timesteps')
	parser.add_argument('--orient_mode', type=int, default=1, help='sites distribution')
	parser.add_argument('--e_mode', type=int, default=1, help='site energy distribution')
	parser.add_argument('--to_gen_normals', type=int, default=1, help='generate input or copy the existing one')
	
	parser.add_argument('--R', type=float, default=4.0, help='R_big')
	parser.add_argument('--site_sgm', type=float, default=1.0, help='site overlaps')
	parser.add_argument('--i_max', type=int, default=-200, help='i_max for site generation')
	parser.add_argument('--epsilon', type=float, default=10.0, help='site-membrane binding')
	parser.add_argument('--n_sites', type=int, default=50, help='N of sites')
	
	parser.add_argument('--no_verbose', type=int, default=1, help='redirect lmp versocity to dev/null')
	parser.add_argument('--to_run_lmp', type=int, default=0, help='launch lammps auto')
	
	clargs = parser.parse_args()
	
	seed = clargs.seed
	timesteps_total = clargs.timesteps_total
	orient_mode = clargs.orient_mode
	e_mode = clargs.e_mode
	n_sites = clargs.n_sites
	R = clargs.R
	site_sgm = clargs.site_sgm
	i_max = clargs.i_max
	epsilon = clargs.epsilon
	
	lib.np_rng = np.random.default_rng(seed)
	
	InputPath = os.path.join(str(os.getcwd()))
	#InputPath = os.path.join(root_path, )
	
	#model_group_name = 'Ort%d_T%d' % (orient_mode, timesteps_total)
	#model_group_name = lib.model_name_fnc(orient_mode, timesteps_total)
	model_group_name = lib.model_name_fnc(R, site_sgm, epsilon, n_sites, timesteps_total, orient_mode, e_mode)
	model_name = os.path.join(model_group_name, 'sd%d' % (seed))
	model_path = os.path.join(lib.root_path, model_name)
	input_data_filepath_orig = InputPath+"/Inputs/InputData.txt"
	input_data_filepath = os.path.join(model_path, 'InputData.txt')
	f_in_filepath = os.path.join(model_path, 'in.local')
	f_dat_filepath = os.path.join(model_path, 'data')
	
	#r = os.mkdir('%s/sd%d'%(InputPath,seed))
	#r = os.mkdir(model_path)
	os.makedirs(model_path, exist_ok=True)
	
	if(clargs.to_gen_normals):
		if(os.path.isfile(input_data_filepath)):
			if(input('file %s existst; Continue? [y/N]') != 'y'):
				return 0
		
		generate_nanoparticle(input_data_filepath, \
							n=n_sites, \
							R=R, \
							site_sgm=site_sgm, \
							i_max=i_max, \
							epsilon=lib.tht_e_fnc_dict[e_mode] if(e_mode >= 0) else epsilon, \
							tht_fnc=lib.tht_fnc_dict[orient_mode])
		
	else:
		lib.run_it('cp %s %s' % (input_data_filepath_orig, input_data_filepath))
	
	print("Reading data from", input_data_filepath)
	f = pandas.read_csv(input_data_filepath, header = [0],sep = ' ')
	IDs = f.ID
	types = f.type
	epslions = f.Epsilon # this is the interaction strength of patches (could be changed to have different values for different patches)
	x,y,z = f.x,f.y,f.z
	NumPatches= len(IDs)
	
	
	#f2 = open(InputPath+"/Inputs/InputData2.txt",'w')
	#f2.write('ID type Epsilon x y z\n')
	#for i in range(len(IDs)):
	#	f2.write(str(IDs[i])+' '+str(types[i])+' '+str(epslions[i])+' '+str(x[i])+' '+str(y[i])+' '+str(float(z[i]-6.5))+'\n')
	#f2.close()
	
	print("Writing input file in", f_in_filepath)
	
	f_in = open(f_in_filepath, 'w') # this is our input files directory
	f_in.write('''
	
# set up our simulation environment
dimension		3
units			lj
atom_style		hybrid sphere dipole
boundary		p p p\n''')
	f_in.write('read_data			"%s"\n'%(f_dat_filepath)) # here we specify where the data is for the initial configurations of our particles
	f_in.write('''
	#group particles according to their types

group		mem			type 1
group		vehicle		type 2
group		ligand 		type >= 3
group		np	  	type >= 2


#give our particle a small kick towards the membrane

velocity	np	set 0 0 -2

#membrane parameters (see Yuan 2011)
variable	rc			equal	2.6
variable	rmin		equal	1.122462
variable	rclus		equal	2.0
variable	mu			equal	3
variable	zeta		equal	4
variable	eps			equal	4.34
variable	sigma		equal	1.00
variable	theta0_11	equal	0
variable 	memsize equal "count(mem)"

# nanoparticle parameters

variable	peps		equal	2.2

# set up additional variables to be computed during the simulation (see lammps wiki for full explainations)

#compute 		1 		all 		pair lj/cut epair
compute 		cls 	all 		cluster/atom ${rclus}
compute 		ct 		all 		temp/sphere
compute_modify 	ct 		extra 		${memsize}

# set up the pair style for the membrane

pair_style	hybrid		membrane ${rc}	lj/cut 5.04

# membrane-nanoparticle interactions, each is set separately so any distribution of ligand strengths is possible

pair_coeff		*	*	lj/cut		0.0				0.0	0.0
pair_coeff		1	2	lj/cut		100				4.0	4.45
''')
	for n in range(NumPatches):
		f_in.write('pair_coeff		1	'+str(types[n])+'	lj/cut		'+str(epslions[n])+'			1	1.8\n')
	
	f_in.write('''#we set the interaction to zero at its cutoff distance, otherwise we will have a discontinuity

pair_modify		pair 	lj/cut	shift yes

#membrane-membrane interactions (these can be changed by messing with the variables at the top)

pair_coeff		1	1	membrane ${eps} ${sigma} ${rmin} ${rc} ${zeta} ${mu} ${theta0_11}

neigh_modify	every 1	delay 1	exclude group np np

# we set up the integration parameters
''')
	
	f_in.write('fix			fLANG		all		langevin 1.0 1.0 1.0 '+str(seed)+' zero yes omega yes\n') ## reading seed!
	f_in.write('''
fix			fRIGID		np		rigid/nve	group 1 np
fix			fNPH		mem		nph/sphere	x 0.0 0.0 10.0	y 0.0 0.0 10.0 couple xy update dipole dilate all
fix_modify	fNPH		temp ct press thermo_press

#output settings, changing peps will change the output file name as well, change this by removing ${peps} from the dump file name
''')

	f_in.write('dump			coords all custom 100 output.xyz id type x y z c_cls')
	f_in.write('''  
dump_modify	coords sort id

thermo_style	custom	step pe ke etotal

# set up our timestep and runtime

timestep	   0.01
thermo		 100
run			%d''' % (timesteps_total)) ### IF you want to change the length of the simulation, do it here
		
	f_in.close()
			   
		


	print("Writing data file", f_dat_filepath)
	
	
	f_dat = open(f_dat_filepath,'w')

	# Reading the original data file
	#f_ogdat = open(InputPath+"/Inputs//Membrane.data")
	f_ogdat = open(InputPath+"/Inputs/Membrane.data")
	lines = f_ogdat.readlines()
	f_ogdat.close()

	# Box lines
	blines = lines[7:10]
	# Membrane positions lines
	mlines = lines[41:2944]

	# Writing the data file
	f_dat.write('LAMMPS data file\n\n')
	f_dat.write(str(2901+NumPatches)+' atoms\n')
	f_dat.write(str(2+NumPatches)+' atom types\n')
	for line in blines:
		f_dat.write(line)
	f_dat.write('\n')
	f_dat.write('Masses\n\n')
	for i in range(NumPatches+2):
		f_dat.write(str(i+1)+' 1\n')
	f_dat.write('\n')
	for line in mlines:
		f_dat.write(line)
	for n in range(NumPatches):
		f_dat.write(str(2902+n)+ ' ' + str(types[n])+' ' +str(x[n])+' ' +str(y[n])+ ' ' +str(z[n]+6.5)+'  1 1 0   0 0 0\n') ## Inputing postitions of patches
	f_dat.close()
		
		
	
	if(clargs.to_run_lmp):
		os.chdir(model_path)
		#lib.run_it('lmp_della_2022_3_CV_dipole_ser_my -in in.local %s' % (' &> /dev/null &' if(clargs.no_verbose) else ''))
		lib.run_it('lmp_lmp_serial_BSS24 -in in.local %s' % (' &> /dev/null &' if(clargs.no_verbose) else ''))
  


if __name__ == "__main__":
	main()
	
	
