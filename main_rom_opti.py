#######################################################################################################################
#######################################################################################################################
# This file is part of the paper "Optimality-Based Control Space Reduction for Infinite-Dimensional Control Spaces"
# by Michael Kartmann, Stefan Volkwein
## Author of this code: Michael Kartmann
#######################################################################################################################
#######################################################################################################################

from discretizer import discretize_stability_problem
import numpy as np
from reductor import pod_reductor

# %%

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
Nx = 100  # 100 #20
fom = discretize_stability_problem(T=T, Nx=Nx, K=K)
print('FOM constructed')
fom.print_info()
fom.update_cost_data(Yd=fom.cost_data.Yd,
                     YT=fom.cost_data.Yd[:, -1],
                     Ud=np.zeros((fom.input_dim, fom.time_disc.K)),
                     weights=[1, beta, 0],
                     input_product=fom.cost_data.input_product,
                     output_product=fom.cost_data.output_product)

# get output trajectory and visualize it
if 0:
    yd_norm = fom.space_norm_trajectory(fom.cost_data.Yd, norm='output')
    fom.visualize_1d(yd_norm, title=r'Output norm $y_d$', semi=True)

# %% setup optimization

#### optimization options
# BB solver options
tol = 1e-12
maxit = 400

# for computing exact solution
optionsBB_exact = fom.set_default_options(tol=tol,
                                          maxit=maxit,
                                          save=False,
                                          plot=False,
                                          print_info=True)

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
# %% create roms

# generate snapshots
U_0 = np.zeros((fom.input_dim, fom.time_disc.K))
_, Y_snap, P_snap, _, _, _ = fom.optimal_control_est(Ur=U_0)
Snapshots_POD = [Y_snap, P_snap]

# construct ROM
rom = r_pod_no_c.get_rom(l=l_POD,
                         Snapshots=Snapshots_POD,
                         space_product=fom.pde.products['H1'],
                         time_product=fom.time_disc.D_diag,
                         PODmethod=0,
                         plot=True
                         )
# construct FULLROM
fullrom = r_pod.get_rom(l=l_POD,
                        Snapshots=Snapshots_POD,
                        space_product=fom.pde.products['H1'],
                        time_product=fom.time_disc.D_diag,
                        PODmethod=0,
                        plot=True,
                        pod_basis=r_pod_no_c.POD_Basis,
                        pod_values=r_pod_no_c.POD_values
                        )

# %% ROM optimization

# run both ROM optimizations
print('ROM optimization ...')
u_rom, history_rom = rom.solve_ocp(U_0,
                                   options=optionsBB_exact)

print('Full-ROM optimization ...')
u_fullrom, history_fullrom = fullrom.solve_ocp(r_pod.FOMtoROM(U_0),
                                               options=optionsBB_exact)

# compare errors
e_u = fom.space_time_norm(r_pod.ROMtoFOM(history_fullrom['U_opt']) - history_rom['U_opt'], space_norm="control")
e_y = fom.space_time_norm(r_pod.ROMtoFOM(history_fullrom['Y_opt']) - r_pod.ROMtoFOM(history_rom['Y_opt']),
                          space_norm="L2")
e_p = fom.space_time_norm(r_pod.ROMtoFOM(history_fullrom['P_opt']) - r_pod.ROMtoFOM(history_rom['P_opt']),
                          space_norm="L2")
print(f'Results')
print(f'time rom: {history_rom["time"]}')
print(f'time control-rom: {history_fullrom["time"]}')
print(f'speed-up: {history_rom["time"] / history_fullrom["time"]}')
print(f'error control: {e_u}, L2-error state: {e_y}, L2-error adjoint: {e_p}')
print('-------------------------------------------------------------------')