import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import glob
import re
#import pickle
import shutil
import scipy
import scipy.signal
import copy
import time

np_rng = None
error_str = '!!ERROR OCCURED!!'

font_mode = 'present'
lbl_fontsize_dict = {'work' : plt.rcParams['font.size'], 'present' : 16}
ticks_fontsize_dict = {'work' : plt.rcParams['font.size'], 'present' : 16}

user_home_path = os.path.expanduser('~')
username = user_home_path.split('/')[-1]

root_path = '/scratch/gpfs/yp1065/BSS24/week1'

tht_fnc_dict = \
	{1 : lambda t: np.ones(t.shape) if(isinstance(t, np.ndarray)) else 1.0, \
	2 : lambda t: np.sin(t), \
	3 : lambda t: np.cos((np.pi-t)/2), \
	5 : lambda t: np.cos((np.pi-t)/2)**3, \
	6 : lambda t: np.cos(t/2), \
	8 : lambda t: np.cos(t/2)**2, \
	9 : lambda t: np.cos(t/2)**3, \
	4 : lambda t: np.cos((np.pi-t)/2)**2, \
	7 : lambda t: np.cos(t)**2}

tht_e_fnc_dict = \
	{1 : lambda t: np.ones(t.shape) if(isinstance(t, np.ndarray)) else 1.0, \
	2 : lambda t: np.sin(t), \
	3 : lambda t: np.cos((np.pi-t)/2), \
	5 : lambda t: np.cos((np.pi-t)/2)**3, \
	6 : lambda t: np.cos(t/2), \
	8 : lambda t: np.cos(t/2)**2, \
	9 : lambda t: np.cos(t/2)**3, \
	4 : lambda t: np.cos((np.pi-t)/2)**2, \
	7 : lambda t: np.cos(t)**2, \
	10 : lambda t: np.cos(t)}
# 2 3 6 10

model_name_fnc = lambda R, site_sgm, epsilon, n_sites, timesteps_total, orient_mode, e_mode: \
	'R%s_sgm%s_eps%s_Ns%d_T%d_Ort%d_eMode%d' % (f2s(R), f2s(site_sgm), f2s(epsilon), n_sites, timesteps_total, orient_mode, e_mode)


def find_key(keys, key0):
	for _key in keys:
		if(_key == key0):
			return 1

	return 0

def safe_copy(src, dst):
	if(src != dst):
		if(os.path.isfile(dst)):
			os.remove(dst)
		else:
			os.makedirs(os.path.dirname(dst), exist_ok=True)
		shutil.copy2(src, dst)

def git_root_path():
	try:
		return run_it(['git', 'rev-parse', '--show-toplevel'], check=True, verbose=False)[:-1]
	except:
		return run_it(['pwd'], check=True, verbose=False)[:-1]

def errorbar_str(x, dx, nd0=2, pm_str=' \pm '):
	return pm_str.join(list(errorbar_strs(x, dx, nd0=nd0)))
	# nd = int(np.round(max(1, np.log10(np.abs(x) / dx)))) + 2
	# dx_nd = 1 if(eps_err(float(f2s(dx, 2)), float(f2s(dx, 1))) < 0.2) else 2
	# return f2s(x, nd) + ' \pm ' + f2s(dx, dx_nd)

def errorbar_strs(x, dx, nd0=2):
	if(dx is None):
		return 'None', 'None'
	else:
		if(dx > 0):
			nd = int(np.round(max(1, np.log10(np.abs(x) / dx)))) + nd0
			dx_nd = nd0 - (1 if(eps_err(float(f2s(dx, nd0)), float(f2s(dx, nd0-1))) < 0.2 * 10**(2-nd0)) else 0)
		else:
			nd = nd0
			dx_nd = 1

		return f2s(x, nd), f2s(dx, dx_nd) if(dx > 0) else '0'

def run_it(command, shell=False, verbose=True, check=False, to_run=True):
	if(isinstance(command, str)):
		if(' ' in command):
			shell = True

	if(verbose):
		print(command)

	if(to_run):
		res = subprocess.check_output(command, shell=shell).decode('utf-8')  if(check) else subprocess.run(command, shell=shell)

	return res if(to_run) else command

def f2s(x, n=3):
	return '%s' % float(('%.' + str(n) + 'g') % x)

def join_lbls(arr, spr, n=3):
	return spr.join([f2s(a, n=n) for a in arr])

def find_in_files(s, check=False, verbose=True, flags='-nHIrF'):
	return run_it('grep %s -- %s' % (flags, s), check=check, verbose=verbose)

def eps_err(a, b, my_eps=np.finfo(float).tiny): # exp(|ln(a/b)|) - 1
	x_min = min(abs(a),abs(b))
	if(x_min < my_eps):
		if(max(abs(a),abs(b)) < my_eps):
			return 0
		else:
			return 1/my_eps
	else:
		return abs(a - b)/x_min

def resource_usage_point_report(point_name=""):
	usage = resource.getrusage(resource.RUSAGE_SELF)
	return r'%s: usertime=%s systime=%s mem=%s mb' % (point_name, usage[0], usage[1], usage[2]/1024.0)

def memory_usage_report_UNIX():
	total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
	return total_memory / 1024, used_memory / 1024, free_memory / 1024

def add_legend(fig, ax, do_tight=True, do_legend=True, loc='best', font_size=None):
	if(do_legend):
		ax.legend(prop={"size" : lbl_fontsize_dict[font_mode] if(font_size is None) else (lbl_fontsize_dict[font_size] if(isinstance(font_size, str)) else font_size)}, loc=loc)
	if(do_tight):
		fig.tight_layout()

def get_fig(xlbl, ylbl, title=None, xscl='linear', yscl='linear', \
			xlbl_fontsize=None, ylbl_fontsize=None, title_fontsize=None, \
			tick_major_fontsize=None, tick_minor_fontsize=None, \
			projection=None, zscl='linear', zlbl='z', zlbl_fontsize=None):
	if(title is None):
		title = (ylbl + '(' + xlbl + ')') if(projection is None) else ('%s(%s, %s)' % (zlbl, xlbl, ylbl))

	if(xlbl_fontsize is None):
		xlbl_fontsize = lbl_fontsize_dict[font_mode]
	if(ylbl_fontsize is None):
		ylbl_fontsize = lbl_fontsize_dict[font_mode]
	if(title_fontsize is None):
		title_fontsize = lbl_fontsize_dict[font_mode]
	if(tick_major_fontsize is None):
		tick_major_fontsize = ticks_fontsize_dict[font_mode]

	#fig = plt.figure() if(fig_num is None) else plt.figure(fig_num)
	fig = plt.figure()
	fig_num = plt.gcf().number
	ax = fig.add_subplot(projection=projection)

	fig.suptitle(title, fontsize=title_fontsize)
	#ax.set_title(title, fontsize=title_fontsize)
	ax.set_xlabel(xlbl, fontsize=xlbl_fontsize)
	ax.set_ylabel(ylbl, fontsize=ylbl_fontsize)
	ax.set_xscale(xscl)
	ax.set_yscale(yscl)
	ax.tick_params(axis='both', which='major', labelsize=tick_major_fontsize)
	if(tick_minor_fontsize is not None):
		ax.tick_params(axis='both', which='minor', labelsize=tick_minor_fontsize)

	if(projection == '3d'):
		if(zlbl_fontsize is None):
			zlbl_fontsize = lbl_fontsize_dict[font_mode]
		ax.set_zscale(zscl)
		ax.set_zlabel(zlbl, fontsize=zlbl_fontsize)
		if(np.any(np.array([(s != 'linear') for s in [xscl, yscl, zscl]]))):
			print('WARNING: matplotlib does not support non-linear scaling in 3D')

	return fig, ax, fig_num


