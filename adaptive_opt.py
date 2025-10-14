#######################################################################################################################
#######################################################################################################################
# This file is part of the paper "Optimality-Based Control Space Reduction for Infinite-Dimensional Control Spaces"
# by Michael Kartmann, Stefan Volkwein
## Author of this code: Michael Kartmann
#######################################################################################################################
#######################################################################################################################

from time import time


def adaptive_optimization(fom, reductor, U0, l_POD, options, control_reduced=True):
    print('-------------------------------------------------------------------')
    print("Starting Adaptive Optimization ")
    print(f"control_reduced = {control_reduced}, tol = {options['tol']} and tol_est = {options['est_tol']}")
    print('-------------------------------------------------------------------')

    t = time()

    # eval error est/gradient
    U = U0
    est, Y, P, _, dJ, grad_norm = fom.optimal_control_est(Ur=U)
    lower_est = grad_norm / (fom.cost_data.weights[1] + fom.pde.constants['CS**2'])
    grad_norm0 = grad_norm
    J_k = fom.J(U, Y=Y, output=Y)
    J_kold = J_k
    k = 0
    Snapshots = []
    rom = None
    history = {
        'gradient_norm': [grad_norm],
        'lower_est': [lower_est],
        'true_error': [],
        'upper_est': [est],
        'basis_size': [],
        'u_k': [U0],
        'J_k': [J_k],
    }

    while grad_norm > max(options['tol'], options['tol'] * grad_norm0) and k < options['maxit']:

        # print
        print(f"k: {k:2}, grad_norm = {grad_norm: 2.4e}, rel grad_norm = {grad_norm / grad_norm0: 2.4e}, lower bound: {lower_est: 2.4e}, upper bound = {est: 2.4e}, est_tol: {options['est_tol']}, J_k: {J_k: 2.6e}, Diff J_k: {J_k - J_kold: 2.4e}")

        # select snapshots
        if 1:
            Snapshots += [Y, P]
            Snapshots_POD = Snapshots
        else:
            # descent based
            # c = 0.1
            Snapshots += [Y, P]
            print(f'Crit: {J_k}<{J_kold - c * grad_norm}, ')
            if J_k < J_kold - c * grad_norm:
                print('Big step, snapshot set reset')
                Snapshots_POD = [Y, P]
            else:
                print('Small step, collect snapshots')
                Snapshots_POD = Snapshots[-4:]

        # creat rom
        rom = reductor.get_rom(l=l_POD,
                               Snapshots=Snapshots_POD,
                               space_product=fom.pde.products['H1'],
                               time_product=fom.time_disc.D_diag,
                               PODmethod=0,
                               plot=False
                               )

        # project, solve BB, reconstruct
        if control_reduced:
            Uproj = reductor.FOMtoROM(U)
        else:
            Uproj = U
        Unext_proj, hist_inner = rom.solve_ocp(Uproj,
                                               options=options['optionsBB'])
        if control_reduced:
            U = reductor.ROMtoFOM(Unext_proj)
        else:
            U = Unext_proj

        # # save snapshots
        # Yold = Y
        # Pold = P

        # eval error est/gradient
        est, Y, P, _, dJ, grad_norm = fom.optimal_control_est(Ur=U)
        J_kold = J_k
        J_k = fom.J(U, Y=Y, output=Y)
        lower_est = grad_norm / (fom.cost_data.weights[1] + fom.pde.constants['CS**2'])

        # update history
        history['upper_est'].append(est)
        history['basis_size'].append(rom.state_dim)
        history['lower_est'].append(lower_est)
        history['gradient_norm'].append(grad_norm)
        history['u_k'].append(U)
        history['J_k'].append(J_k)

        # update
        k += 1

    print(
        f"k: {k:2}, grad_norm = {grad_norm: 2.4e}, rel grad_norm = {grad_norm / grad_norm0: 2.4e}, lower bound: {lower_est: 2.4e}, upper bound = {est: 2.4e}, est_tol: {options['est_tol']}, J_k: {J_k: 2.6e}, Diff J_k: {J_k - J_kold: 2.4e}")
    # finalize
    history['Y_opt'] = Y
    history['P_opt'] = P
    history['U_opt'] = U
    history['time'] = time() - t
    history['est'] = est
    history['grad_norm'] = grad_norm
    history['k'] = k
    if k == options['maxit']:
        history[
            'flag'] = f'ADAPTIV OPT reached maxit of k = {k:2} iterations in {history["time"]: .3f} seconds with gradient norm of {grad_norm:2.4e}, rel grad_norm = {grad_norm / grad_norm0: 2.4e} and est = {est: 2.4e}.'
    else:
        history[
            'flag'] = f'ADAPTIV OPT converged in k = {k:2} iterations in {history["time"]: .3f} seconds with gradient norm of {grad_norm:2.4e}, rel grad_norm = {grad_norm / grad_norm0: 2.4e} and est = {est: 2.4e}.'
    print(history['flag'])
    print('#############################################################')

    return U, history, rom
