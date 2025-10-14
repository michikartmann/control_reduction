#######################################################################################################################
#######################################################################################################################
# This file is part of the paper "Optimality-Based Control Space Reduction for Infinite-Dimensional Control Spaces"
# by Michael Kartmann, Stefan Volkwein
## Author of this code: Michael Kartmann
#######################################################################################################################
#######################################################################################################################

from discretizer import discretize_stability_problem
from methods import print_results, plot_results
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10})
plt.rcParams['lines.linewidth'] = 2

#%% folders

data_folder = 'data/'
plot_folder = 'plots/'

#%% setup FOM

# get pde model (make sure the discretization setup here coincides with the main file)
T = 1  
K = 101
Nx = 100
fom = discretize_stability_problem(T = T, Nx = Nx, K = K)

#%% read data

import pickle

# beta = 0.1
with open('data/results/run_dim10201_K101_beta0.1/fom.pkl', 'rb') as f:
    history_fom1 = pickle.load(f)
with open('data/results/run_dim10201_K101_beta0.1/fom_exact.pkl', 'rb') as f:
    history_exact1 = pickle.load(f)
with open('data/results/run_dim10201_K101_beta0.1/rom.pkl', 'rb') as f:
    history_rom1 = pickle.load(f)
with open('data/results/run_dim10201_K101_beta0.1/full_rom.pkl', 'rb') as f:
    history_full_rom1 = pickle.load(f)
    
# beta = 0.01
with open('data/results/run_dim10201_K101_beta0.01/fom.pkl', 'rb') as f:
    history_fom2 = pickle.load(f)
with open('data/results/run_dim10201_K101_beta0.01/fom_exact.pkl', 'rb') as f:
    history_exact2 = pickle.load(f)
with open('data/results/run_dim10201_K101_beta0.01/rom.pkl', 'rb') as f:
    history_rom2 = pickle.load(f)
with open('data/results/run_dim10201_K101_beta0.01/full_rom.pkl', 'rb') as f:
    history_full_rom2 = pickle.load(f)
    
# beta = 0.001
with open('data/results/run_dim10201_K101_beta0.001/fom.pkl', 'rb') as f:
    history_fom3 = pickle.load(f)
with open('data/results/run_dim10201_K101_beta0.001/fom_exact.pkl', 'rb') as f:
    history_exact3 = pickle.load(f)
with open('data/results/run_dim10201_K101_beta0.001/rom.pkl', 'rb') as f:
    history_rom3 = pickle.load(f)
with open('data/results/run_dim10201_K101_beta0.001/full_rom.pkl', 'rb') as f:
    history_full_rom3 = pickle.load(f)

# beta = 0.0001
with open('data/results/run_dim10201_K101_beta0.0001/fom.pkl', 'rb') as f:
    history_fom4 = pickle.load(f)
with open('data/results/run_dim10201_K101_beta0.0001/fom_exact.pkl', 'rb') as f:
    history_exact4 = pickle.load(f)
with open('data/results/run_dim10201_K101_beta0.0001/rom.pkl', 'rb') as f:
    history_rom4 = pickle.load(f)
with open('data/results/run_dim10201_K101_beta0.0001/full_rom.pkl', 'rb') as f:
    history_full_rom4 = pickle.load(f)

h_fom = [history_fom1, history_fom2, history_fom3, history_fom4]
h_exact = [history_exact1, history_exact2, history_exact3, history_exact4]
h_rom = [history_rom1, history_rom2, history_rom3, history_rom4]
h_full_rom = [history_full_rom1, history_full_rom2, history_full_rom3, history_full_rom4]

#%% print and plot results

Beta = [1e-1,1e-2,1e-3,1e-4]

for i in range(len(h_fom)):
    print(f'beta = {Beta[i]}--------------------------------------------------')
    print_results(fom, h_fom[i], h_rom[i], h_full_rom[i], h_exact[i])
    print('-------------------------------------------------------------------')

#%% plot results

for i in range(len(h_fom)):
     plot_results(fom, h_fom[i], h_rom[i], h_full_rom[i], h_exact[i], plot_folder, Beta[i])