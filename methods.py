#######################################################################################################################
#######################################################################################################################
# This file is part of the paper "Optimality-Based Control Space Reduction for Infinite-Dimensional Control Spaces"
# by Michael Kartmann, Stefan Volkwein
## Author of this code: Michael Kartmann
#######################################################################################################################
#######################################################################################################################

import sys


class Collection:
    pass


class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, "w")
        self.console = sys.__stdout__

    def write(self, text):
        self.console.write(text)
        self.file.write(text)

    def flush(self):
        self.console.flush()
        self.file.flush()


def print_results(fom, history_fom, history_rom, history_fullrom, history_exact):
    print(f'FOM ------------- k: {history_fom["k"]} --------------------------')
    print(f'FOM time: {history_fom["time"]: .3f}')
    true_error = fom.space_time_norm(history_exact['U_opt'] - history_fom['U_opt'], space_norm="control")
    print(f'FOM: true error: {true_error: 2.4e}')

    print(f'ROM ------------- k: {history_rom["k"]}---------------------------')
    print(f'ROM: time: {history_rom["time"]: .3f}, speed-up: {history_fom["time"] / history_rom["time"]: .3f}')
    true_error = fom.space_time_norm(history_exact['U_opt'] - history_rom['U_opt'], space_norm="control")
    print(
        f'ROM: lower est: {history_rom["lower_est"][-1]: 2.4e}, true error: {true_error: 2.4e}, est: {history_rom["est"]: 2.4e}, grad_norm: {history_rom["grad_norm"]: 2.4e}, est true?: {history_rom["lower_est"][-1] <= true_error <= history_rom["est"]}')
    print(f'Error ROM - FOM: {fom.space_time_norm(history_rom["U_opt"] - history_fom["U_opt"])}')

    print(f'Full-ROM -------- k: {history_fullrom["k"]}------------------------')
    print(
        f'FULL-ROM: time: {history_fullrom["time"]: .3f}, speed-up: {history_fom["time"] / history_fullrom["time"]: .3f}')
    true_error_c = fom.space_time_norm(history_exact['U_opt'] - history_fullrom['U_opt'], space_norm="control")
    print(
        f'FULL-ROM: lower est: {history_fullrom["lower_est"][-1]: 2.4e}, true error: {true_error_c: 2.4e}, est: {history_fullrom["est"]: 2.4e}, grad_norm: {history_fullrom["grad_norm"]: 2.4e}, est true?: {history_fullrom["lower_est"][-1] <= true_error <= history_fullrom["est"]}')
    print(f'Error Full-ROM - ROM: {fom.space_time_norm(history_rom["U_opt"] - history_fullrom["U_opt"])}')


def plot_results(fom, history_fom, history_rom, history_fullrom, history_exact, plot_folder, beta, fontsize=15,
                 plot_solution=True):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{accents}'
    plt.rcParams.update({'font.size': 15})
    plt.rcParams['lines.linewidth'] = 3

    #### plots of state and control
    if plot_solution:
        k_set = [10, 50, 80]

        # target
        for kk in k_set:
            path = plot_folder + fr'target_dim{fom.state_dim}_K{fom.time_disc.K}_k{kk}.eps'
            fom.plot_3d(fom.cost_data.Yd[:, kk], title=fr'$y_d$ $t$ = {fom.time_disc.t_v[kk]}', path=path)

        #### FOM
        # FOM state
        for kk in k_set:
            path = plot_folder + fr'fom_state_dim{fom.state_dim}_K{fom.time_disc.K}_beta{beta}_k{kk}.eps'
            fom.plot_3d(history_fom['Y_opt'][:, kk], title=fr'FOM state $t$ = {fom.time_disc.t_v[kk]}', path=path)

        # FOM control
        for kk in k_set:
            path = plot_folder + fr'fom_control_dim{fom.state_dim}_K{fom.time_disc.K}_beta{beta}_k{kk}.eps'
            fom.plot_3d(history_fom['U_opt'][:, kk], title=fr'FOM control $t$ = {fom.time_disc.t_v[kk]}', path=path)

        #### ROM
        # ROM state
        for kk in k_set:
            path = plot_folder + fr'rom_state_dim{fom.state_dim}_K{fom.time_disc.K}_beta{beta}_k{kk}.eps'
            fom.plot_3d(history_rom['Y_opt'][:, kk], title=fr'ROM state $t$ = {fom.time_disc.t_v[kk]}', path=path)

        # ROM control
        for kk in k_set:
            path = plot_folder + fr'rom_control_dim{fom.state_dim}_K{fom.time_disc.K}_beta{beta}_k{kk}.eps'
            fom.plot_3d(history_rom['U_opt'][:, kk], title=fr'ROM control $t$ = {fom.time_disc.t_v[kk]}', path=path)

        #### FULL-ROM
        # Full-ROM state
        for kk in k_set:
            path = plot_folder + fr'fullrom_state_dim{fom.state_dim}_K{fom.time_disc.K}_beta{beta}_k{kk}.eps'
            fom.plot_3d(history_fullrom['Y_opt'][:, kk], title=fr'Full-ROM state $t$ = {fom.time_disc.t_v[kk]}',
                        path=path)

        # Full-ROM state
        for kk in k_set:
            path = plot_folder + fr'fullrom_control_dim{fom.state_dim}_K{fom.time_disc.K}_beta{beta}_k{kk}.eps'
            fom.plot_3d(history_fullrom['U_opt'][:, kk], title=fr'Full-ROM control $t$ = {fom.time_disc.t_v[kk]}',
                        path=path)

        # norms
        fom.visualize_1d_many([fom.space_norm_trajectory(fom.cost_data.Yd, norm='output'),
                               fom.space_norm_trajectory(history_fom['Y_opt'], norm='output'),
                               fom.space_norm_trajectory(history_fullrom['Y_opt'], norm='output'),
                               fom.space_norm_trajectory(history_rom['Y_opt'], norm='output')
                               ],
                              strings=[r'$y_d$', 'FOM', 'Full-ROM', 'ROM'], semi=True, title='State norm')

        fom.visualize_1d_many([
            fom.space_norm_trajectory(history_fom['U_opt'], norm='control'),
            fom.space_norm_trajectory(history_fullrom['U_opt'], norm='control'),
            fom.space_norm_trajectory(history_rom['U_opt'], norm='control')
        ],
            strings=['FOM', 'Full-ROM', 'ROM'], semi=True, title='Control norm')

    #### convergence plots
    plt.figure()
    plt.semilogy(history_fom['grad_norm'], label='grad norm')
    plt.title(r'FOM', fontsize=fontsize)
    plt.legend()
    plt.grid(True)
    plt.xlabel(r'$k$', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(plot_folder + f"fom_bounds_beta{beta}.eps", format="eps", bbox_inches="tight")
    plt.show()

    u_exact = history_exact['U_opt']
    true_norm_nc = []
    for u_k in history_rom['u_k']:
        true_norm_nc.append(fom.space_time_norm(u_exact - u_k, 'control'))
    plt.figure()
    plt.semilogy(history_rom['gradient_norm'], label=r'$\nabla J(u_k)$', linestyle='-.', marker='D')
    plt.semilogy(history_rom['upper_est'], label=r'$\bar \Delta(u_k)$', linestyle='--', marker='o')
    plt.semilogy(history_rom['lower_est'], label=r'$\underaccent{\bar}{\Delta}(u_k)$', linestyle=':', marker='x')
    plt.semilogy(true_norm_nc, label=r'$e(u_k)$', linestyle='-', marker='^')
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.title(r'ROM', fontsize=fontsize)
    plt.xlabel(r'$k$', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(plot_folder + f"rom_bounds_beta{beta}.eps", format="eps", bbox_inches="tight")
    plt.show()

    true_norm_c = []
    for u_k in history_fullrom['u_k']:
        true_norm_c.append(fom.space_time_norm(u_exact - u_k, 'control'))
    plt.figure()
    plt.semilogy(history_fullrom['gradient_norm'], label=r'$\nabla J(u_k)$', linestyle='-.', marker='D')
    plt.semilogy(history_fullrom['upper_est'], label=r'$\bar \Delta(u_k)$', linestyle='--', marker='o')
    plt.semilogy(history_fullrom['lower_est'], label=r'$\underaccent{\bar}{\Delta}(u_k)$', linestyle=':', marker='x')
    plt.semilogy(true_norm_c, label=r'$e(u_k)$', linestyle='-', marker='^')
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.title(r'Full-ROM', fontsize=fontsize)
    plt.xlabel(r'$k$', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(plot_folder + f"fullrom_bounds_beta{beta}.eps", format="eps", bbox_inches="tight")
    plt.show()

    # basis size against iter
    plt.figure()
    plt.plot(history_fullrom['basis_size'], label='Full-ROM', linestyle='-', marker='o')
    plt.plot(history_rom['basis_size'], label='ROM', linestyle=':', marker='x')
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.title(r'Basis size', fontsize=fontsize)
    plt.xlabel(r'$k$', fontsize=fontsize)
    plt.ylabel(r'$r$', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(plot_folder + f"basissize_beta{beta}.eps", format="eps", bbox_inches="tight")
    plt.show()
