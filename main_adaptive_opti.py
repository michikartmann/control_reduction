#######################################################################################################################
#######################################################################################################################
# This file is part of the paper "Optimality-Based Control Space Reduction for Infinite-Dimensional Control Spaces"
# by Michael Kartmann, Stefan Volkwein
## Author of this code: Michael Kartmann
#######################################################################################################################
#######################################################################################################################

from discretizer import discretize_stability_problem
import numpy as np
from methods import print_results, plot_results, Tee
from adaptive_opt import adaptive_optimization
from reductor import pod_reductor
import sys
import pickle
import os

# %% flags

# vary beta to run experiments
beta = 1e-1
# beta = 1e-2
# beta = 1e-3
# beta = 1e-4

# folders and flags
data_folder = 'data/'
plot_folder = 'plots/'
derivativecheck = False

# %% setup FOM

# get pde model
T = 1
K = 101
Nx = 100  # 20
fom = discretize_stability_problem(T=T, Nx=Nx, K=K)
print('FOM constructed')
fom.print_info()

# get target yd
fom.update_cost_data(Yd=fom.cost_data.Yd,
                     YT=fom.cost_data.Yd[:, -1],
                     Ud=np.zeros((fom.input_dim, fom.time_disc.K)),
                     weights=[1, beta, 0],
                     input_product=fom.cost_data.input_product,
                     output_product=fom.cost_data.output_product)

# get the output trajectory and visualize it
if 0:
    yd_norm = fom.space_norm_trajectory(fom.cost_data.Yd, norm='output')
    fom.visualize_1d(yd_norm, title=r'Output norm $y_d$', semi=True)

# %% setup optimization

#### optimization options

# starting value U_0
U_0 = np.zeros((fom.input_dim, fom.time_disc.K))

# BB solver options
tol = 1e-8
maxit = 400

# BB optimization options
optionsBB = fom.set_default_options(tol=tol,
                                    maxit=maxit,
                                    save=False,
                                    plot=False,
                                    print_info=True,
                                    solve_true=True)

# for computing exact solution
optionsBB_exact = fom.set_default_options(tol=1e-12,
                                          maxit=maxit,
                                          save=False,
                                          plot=False,
                                          print_info=True,
                                          solve_true=True)

# for inner solver
inneroptions = fom.set_default_options(tol=tol,
                                       maxit=maxit,
                                       save=False,
                                       plot=False,
                                       print_info=False,
                                       solve_true=False)

# options for adaptive POD opt
options = {'optionsBB': inneroptions,
           'maxit': 30,
           'tol': tol,
           'est_tol': tol * fom.cost_data.weights[1],
           }

# %% setup POD

#### POD options
# set maximum basis size
l_POD = 100

# reductor object with control reduction
r_pod = pod_reductor(fom,
                     model_toproject=fom,
                     H_prod=fom.pde.products['L2'],
                     space_product=fom.pde.products['H1'],
                     project_control=True
                     )

# reductor object without control reduction
r_pod_no_c = pod_reductor(fom,
                          model_toproject=fom,
                          H_prod=fom.pde.products['L2'],
                          space_product=fom.pde.products['H1'],
                          project_control=False
                          )

# %%  optimization

#### setup
folder = f"data/run_dim{fom.state_dim}_K{fom.time_disc.K}_beta{fom.cost_data.weights[1]}"
filename_print = "console.txt"
filepath = os.path.join(folder, filename_print)
os.makedirs(folder, exist_ok=True)
original_stdout = sys.stdout
sys.stdout = Tee(filepath)

#### FOM optimization
print('FOM optimization ...')
u_fom, history_fom = fom.solve_ocp(U_0,
                                   options=optionsBB)
fom.print_info()
if derivativecheck:
    fom.derivative_check(derivativecheck)
filename = 'fom.pkl'
path = os.path.join(folder, filename)
with open(path, 'wb') as f:
    pickle.dump(history_fom, f)

# compute exact solution
print('Compute FOM exact solution ...')
u_exact, history_exact = fom.solve_ocp(u_fom,
                                       options=optionsBB_exact)
filename = 'fom_exact.pkl'
path = os.path.join(folder, filename)
with open(path, 'wb') as f:
    pickle.dump(history_exact, f)

#### ROM
print('ROM optimization ...')
u_rom, history_rom, rom = adaptive_optimization(fom, r_pod_no_c, U_0, l_POD, options, control_reduced=False)
rom.print_info()
if derivativecheck:
    rom.derivative_check()
filename = 'rom.pkl'
path = os.path.join(folder, filename)
with open(path, 'wb') as f:
    pickle.dump(history_rom, f)

#### Full-ROM
print('Full-ROM optimization ...')
u_fullrom, history_fullrom, fullrom = adaptive_optimization(fom, r_pod, U_0, l_POD, options, control_reduced=True)
fullrom.print_info()
if derivativecheck:
    fullrom.derivative_check()
filename = 'full_rom.pkl'
path = os.path.join(folder, filename)
with open(path, 'wb') as f:
    pickle.dump(history_fullrom, f)

# print results
print_results(fom, history_fom, history_rom, history_fullrom, history_exact)
print('-------------------------------------------------------------------')

# print as usual
sys.stdout = original_stdout

# %% plots

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{accents}'
plt.rcParams.update({'font.size': 10})
plt.rcParams['lines.linewidth'] = 2

#### timings
plt.figure()
plt.title(label='Timings')
categories = ['FOM', 'ROM', 'Full-ROM']
values = [
    history_fom["time"],
    history_rom["time"],
    history_fullrom["time"],
]
plt.bar(categories, values)
plt.xticks(rotation=45, fontsize=12)
plt.ylabel('Time')

#### timings speedup plot
plt.figure()
plt.title(label='Speedups')
categories = ['ROM', 'Full-ROM']
values = [
    history_fom["time"] / history_rom["time"],
    history_fom["time"] / history_fullrom["time"]
]
plt.bar(categories, values)
plt.xticks(rotation=45, fontsize=12)
plt.ylabel('Time')

#### plot results
plot_results(fom, history_fom, history_rom, history_fullrom, history_exact, plot_folder, beta, plot_solution=True)
