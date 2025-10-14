#######################################################################################################################
#######################################################################################################################
# This file is part of the paper "Optimality-Based Control Space Reduction for Infinite-Dimensional Control Spaces"
# by Michael Kartmann, Stefan Volkwein
## Author of this code: Michael Kartmann
#######################################################################################################################
#######################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from time import time, perf_counter
from methods import Collection
import fenics as fenics


class model:

    def __init__(self, pde_data, cost_data, time_disc, space_disc, options=None, error_estimator=None, reductor=None):
        self.pde = pde_data
        self.cost_data = cost_data
        self.options = options
        self.time_disc = time_disc
        self.state_dim = pde_data.state_dim
        self.input_dim = pde_data.input_dim
        self.output_dim = pde_data.output_dim
        self.products = pde_data.products
        self.model_type = pde_data.type
        self.space_disc = space_disc
        self.error_estimator = error_estimator
        self.theta = 1
        self.reductor = reductor
        assert self.theta == 1, 'factorization does only work for theta equal one.... fix this'
        if options is None:
            options = Collection()
            options.factorize = False
            self.pde.factorized_op = None
            self.pde.factorized_op_adjoint = None
            self.options = options

    def print_info(self):
        print(
            f'model name: {self.model_type}, state dim: {self.state_dim}, control dim: {self.input_dim}, output dim: {self.output_dim}, number of time steps: {self.time_disc.K}')

    def isFOM(self):
        return 'FOM' in self.pde.type

    def isSwitchModel(self):
        return 'Switch' in self.pde.type

    def isTimeVaryingModel(self):
        return 'TimeVarying' in self.pde.type

    def isStateDep(self):
        return 'StateDep' in self.pde.type

    def isContinuousControl(self):
        return 'Distributed' in self.pde.type

    def isNotControlReduced(self):
        return '_not_reduced' in self.pde.type

    def update_cost_data(self, Yd=None, YT=None, Ud=None, weights=None, input_product=None, output_product=None,
                         desired_switching_profile=None):
        if input_product is not None:
            self.cost_data.input_product = input_product
        if output_product is not None:
            self.cost_data.output_product = output_product
        if weights is not None:
            self.cost_data.weights = weights
        if Ud is not None:
            if self.isContinuousControl():
                self.cost_data.Ud = Ud
                self.cost_data.Mc_Ud = self.cost_data.input_product.dot(Ud)
                self.cost_data.Ud_Mc_Ud = self.space_product_trajectory(Ud, Ud, norm='control')
            else:
                self.cost_data.Ud = Ud
        if Yd is not None:
            self.cost_data.Yd = Yd
            self.cost_data.Mc_Yd = self.cost_data.output_product @ Yd
            self.cost_data.Yd_Mc_Yd = self.space_product_trajectory(Yd, Yd, norm='output')  # trajectory
        if YT is not None:
            self.cost_data.YT = YT
            self.cost_data.Mc_YT = self.cost_data.output_product @ YT
            self.cost_data.YT_Mc_YT = self.space_product(YT, YT, space_norm='output')  # constant value

        if desired_switching_profile is not None:
            self.cost_data.desired_switching_profile = desired_switching_profile

    def solve_linear_system(self, M, A, dt, theta, rhs):

        # delete dirichlet dofs
        if self.isFOM() and self.space_disc.DirichletBC is not None and 'Dirichlet' in self.model_type:
            _, rhs = self.space_disc.DirichletClearFun(LHS=None, rhs=rhs)

        LHS = M + theta * dt * A
        if self.isFOM():
            out = spsolve(LHS, rhs)
        else:
            out = np.linalg.solve(LHS, rhs)
        return out

    def assemble_matrices_at_time(self, t):
        A = self.pde.A[0] * self.pde.A_time_coefficient[0](t) + self.pde.A[1] * self.pde.A_time_coefficient[1](t) + \
            self.pde.A[2] * self.pde.A_time_coefficient[2](t)
        B = self.pde.B[0] * self.pde.B_time_coefficient(t)
        C = self.pde.C[0] * self.pde.C_time_coefficient(t)
        M = self.pde.M[0] * self.pde.M_time_coefficient(t)
        return A, M, B, C

    def solve_state(self, U, theta=1, print_=False, y0=None):

        start_time = perf_counter()
        F = self.pde.F
        if y0 is None:
            y0 = self.pde.y0
        time_disc = self.time_disc

        # init
        dt = time_disc.dt
        K = time_disc.K
        yy = y0.copy()
        Y = yy.copy().reshape(-1, 1)
        t = time_disc.t_v[0]
        A, M, B, C = self.assemble_matrices_at_time(t)
        out = [C @ yy]
        if print_:
            print(f'k = {0}: t = {t},')

        # time stepping
        for k in range(1, K):

            # get current time
            t = time_disc.t_v[k]
            A, M, B, C = self.assemble_matrices_at_time(t)

            # build LHS and rhs
            RHS_mat = M + (theta - 1) * dt * A
            rhs = RHS_mat.dot(yy)
            if F is not None:
                rhs += dt * (theta * F[:, k] + (1 - theta) * F[:, k - 1])
            if U is not None:
                rhs += dt * B.dot((theta * U[:, k] + (1 - theta) * U[:, k - 1]))

            # solve LS
            yy = self.solve_linear_system(M, A, dt, theta, rhs)

            # save and compute output
            Y = np.concatenate((Y, yy.copy().reshape(-1, 1)), axis=1)
            out.append(C @ yy)
            if print_:
                print(f'k = {k}: t = {t}')

        end_time = perf_counter()
        if print_:
            print(f'Time stepping finished in time {end_time - start_time}')

        return Y, np.array(out).T, end_time

    def solve_adjoint(self, Z, ZT, theta=1):

        time_disc = self.time_disc
        dt = time_disc.dt
        K = time_disc.K
        t = time_disc.t_v[-1]

        ### init
        A, M, B, C = self.assemble_matrices_at_time(t)
        rhs = dt * self.cost_data.weights[0] * C.T @ Z[:, -1]
        rhs += self.cost_data.weights[2] * C.T @ ZT
        p = self.solve_linear_system(M, A.T, dt, theta, rhs)
        P = p.copy().reshape(-1, 1)
        B_listTP = [B.T.dot(p)]

        for k in range(K - 2, -1, -1):
            # get t and mats
            t = time_disc.t_v[k]
            A, M, B, C = self.assemble_matrices_at_time(t=t)

            # assemble rhs and lhs
            F = dt * self.cost_data.weights[0] * C.T @ Z[:, k]
            rhs = M.dot(p) + F

            # solve system
            p = self.solve_linear_system(M, A.T, dt, theta, rhs)

            # append and compute output
            P = np.concatenate((p.reshape(-1, 1), P), axis=1)
            B_listTP.append(B.T.dot(p))

        # reverse time
        B_listTP.reverse()

        return P, np.array(B_listTP).T

    def get_snapshots(self, U, y0=None):
        if y0 is None:
            y0 = self.pde.y0

        if len(U.shape) == 1:
            U = self.vector_to_matrix(U, self.input_dim)
        Y, output, _ = self.solve_state(U=U, theta=self.theta, y0=y0)
        Z = self.cost_data.output_product.dot(output) - self.cost_data.Mc_Yd
        ZT = self.cost_data.output_product.dot(output[:, -1]) - self.cost_data.Mc_YT
        P, B_listTP = self.solve_adjoint(Z, ZT)
        return Y, P, output

    def J(self, u, Y=None, output=None):
        U = self.vector_to_matrix(u, self.input_dim)
        if Y is None or output is None:
            Y, output, time = self.solve_state(U=U, theta=self.theta)

        # quadratic stuff trajectory
        J1_2 = 0.5 * self.cost_data.weights[0] * self.space_time_product(output, output, 'output')
        J1_2 -= self.cost_data.weights[0] * self.space_time_product(output, self.cost_data.Mc_Yd, 'identity')  ##change
        J1_2 += 0.5 * self.cost_data.weights[0] * self.time_norm_scalar(self.cost_data.Yd_Mc_Yd)

        # quadratic stuff end time
        J3_2 = 0.5 * self.cost_data.weights[2] * self.space_product(output[:, -1], output[:, -1], space_norm='output')
        J3_2 -= self.cost_data.weights[2] * self.space_product(output[:, -1], self.cost_data.Mc_YT,
                                                               space_norm='identity')  ###change
        J3_2 += 0.5 * self.cost_data.weights[2] * self.cost_data.YT_Mc_YT

        # quadratic stuff control
        J2_2 = 0.5 * self.cost_data.weights[1] * self.space_time_product(U, U, space_norm='control')
        J2_2 -= self.cost_data.weights[1] * self.space_time_product(U, self.cost_data.Mc_Ud,
                                                                    space_norm='identity')  ###change
        J2_2 += 0.5 * self.cost_data.weights[1] * self.time_norm_scalar(self.cost_data.Ud_Mc_Ud)

        # set together
        J = J1_2 + J2_2 + J3_2

        return J

    def Jtracking_trajectory(self, u, Y=None, output=None):
        U = self.vector_to_matrix(u, self.input_dim)
        if Y is None or output is None:
            Y, output, time = self.solve_state(U=U, theta=self.theta)

        # get tracking trajectory
        J1_2 = 0.5 * self.cost_data.weights[0] * self.space_time_product(output, output, 'output',
                                                                         return_trajectory=True)
        J1_2 -= self.cost_data.weights[0] * self.space_time_product(output, self.cost_data.Mc_Yd, 'identity',
                                                                    return_trajectory=True)  ##change
        J1_2 += 0.5 * self.cost_data.weights[0] * self.cost_data.Yd_Mc_Yd

        # get control term trajectory
        J2_2 = 0.5 * self.cost_data.weights[1] * self.space_time_product(U, U, space_norm='control',
                                                                         return_trajectory=True)
        J2_2 -= self.cost_data.weights[1] * self.space_time_product(U, self.cost_data.Mc_Ud, space_norm='identity',
                                                                    return_trajectory=True)  ###change
        J2_2 += 0.5 * self.cost_data.weights[1] * self.cost_data.Ud_Mc_Ud

        return J1_2, J2_2, J1_2 + J2_2

    def gradJ_OBD(self, u, Y=None, output=None, P=None):
        U = self.vector_to_matrix(u, self.input_dim)
        if Y is None or output is None:
            Y, output, _ = self.solve_state(U=U, theta=self.theta)

        if P is None:
            Z = self.cost_data.output_product.dot(output) - self.cost_data.Mc_Yd
            ZT = self.cost_data.output_product.dot(output[:, -1]) - self.cost_data.Mc_YT
            P, B_listTP = self.solve_adjoint(Z, ZT)

        if not self.isFOM() and self.isNotControlReduced():
            dJ1 = self.reductor.ROMtoFOM(P)
            dJ2 = self.cost_data.weights[1] * (U - self.cost_data.Ud)
            dJ = dJ1 + dJ2
        else:
            dJ1 = P
            dJ2 = self.cost_data.weights[1] * (U - self.cost_data.Ud)
            dJ = dJ1 + dJ2

        return dJ.flatten(), output, Y, P

    # %% products

    def compute_rietzrepresentant(self, y, norm_type='H1dual'):
        if norm_type == 'H1dual':
            if self.isFOM() and self.space_disc.DirichletBC is not None and 'Dirichlet' in self.model_type:
                _, y = self.space_disc.DirichletClearFun(LHS=None, rhs=y)
            y = self.pde.factorizedV1(y)
            return y
        elif norm_type == 'H10dual':
            assert 'Dirichlet' in self.model_type, 'this is no norm for this model'
            if self.isFOM() and self.space_disc.DirichletBC is not None and 'Dirichlet' in self.model_type:
                _, y = self.space_disc.DirichletClearFun(LHS=None, rhs=y)
            y = self.pde.factorizedV2(y)
            return y

    def space_product_trajectory(self, v, w, norm='L2'):
        out = []
        for i in range(v.shape[1]):
            out.append(self.space_product(v[:, i], w[:, i], norm))
        return np.array(out).T

    def space_norm_trajectory(self, v, norm='L2'):
        out = []
        for i in range(v.shape[1]):
            out.append(self.space_norm(v[:, i], norm))
        return out

    def time_norm_scalar(self, V, time_norm=None):
        if time_norm is None:
            time_norm = np.concatenate(([0], self.time_disc.D[1:]))
        return np.vdot(time_norm, V)

    def L2_scalar_norm(self, v):
        return np.sqrt(v.T.dot(self.time_disc.D * v))

    def space_product(self, y1, y2, space_norm='L2'):
        if space_norm == 'L2':
            return y1.T.dot(self.products['L2'].dot(y2))
        elif space_norm == 'H1':
            return y1.T.dot(self.products['H1'].dot(y2))
        elif space_norm == 'H10':
            return y1.T.dot(self.products['H10'].dot(y2))
        elif space_norm == 'output':
            return y1.T.dot(self.cost_data.output_product.dot(y2))
        elif space_norm == 'control':
            return y1.T.dot(self.cost_data.input_product.dot(y2))
        elif space_norm == 'identity':
            return y1.T.dot(y2)
        elif space_norm == 'H1dual':
            y2 = self.compute_rietzrepresentant(y2, norm_type=space_norm)
            return y1.T.dot(y2)
        elif space_norm == 'H10dual':
            y2 = self.compute_rietzrepresentant(y2, norm_type=space_norm)
            return y1.T.dot(y2)

    def space_norm(self, y1, space_norm='L2'):
        return np.sqrt(self.space_product(y1, y1, space_norm))

    def space_time_product(self, v, w, space_norm='L2', time_norm=None, return_trajectory=False, space_mat=None):
        if time_norm is None:
            time_norm = np.concatenate(([0], self.time_disc.D[1:]))
        if space_mat is not None:
            if len(v.shape) < 2:
                v = self.vector_to_matrix(v, self.state_dim)
            if len(w.shape) < 2:
                w = self.vector_to_matrix(w, self.state_dim)
            if return_trajectory:
                return np.diag(v.T.dot(space_mat.dot(w)))
            else:
                return np.vdot(time_norm, np.diag(v.T.dot(space_mat.dot(w))))
        if space_norm == 'L2':
            if len(v.shape) < 2:
                v = self.vector_to_matrix(v, self.state_dim)
            if len(w.shape) < 2:
                w = self.vector_to_matrix(w, self.state_dim)
            if return_trajectory:
                return np.diag(v.T.dot(self.products['L2'].dot(w)))
            else:
                return np.vdot(time_norm, np.diag(v.T.dot(self.products['L2'].dot(w))))
        elif space_norm == 'H1':
            if len(v.shape) < 2:
                v = self.vector_to_matrix(v, self.state_dim)
            if len(w.shape) < 2:
                w = self.vector_to_matrix(w, self.state_dim)
            if return_trajectory:
                return np.diag(v.T.dot(self.products['H1'].dot(w)))
            else:
                return np.vdot(time_norm, np.diag(v.T.dot(self.products['H1'].dot(w))))
        elif space_norm == 'H10':
            if len(v.shape) < 2:
                v = self.vector_to_matrix(v, self.state_dim)
            if len(w.shape) < 2:
                w = self.vector_to_matrix(w, self.state_dim)
            if return_trajectory:
                return np.diag(v.T.dot(self.products['H10'].dot(w)))
            else:
                return np.vdot(time_norm, np.diag(v.T.dot(self.products['H10'].dot(w))))
        elif space_norm == 'output':
            if len(v.shape) < 2:
                v = self.vector_to_matrix(v, self.output_dim)
            if len(w.shape) < 2:
                w = self.vector_to_matrix(w, self.output_dim)
            if return_trajectory:
                return np.diag(v.T.dot(self.cost_data.output_product.dot(w)))
            else:
                return np.vdot(time_norm, np.diag(v.T.dot(self.cost_data.output_product.dot(w))))
        elif space_norm == 'control':
            if len(v.shape) < 2:
                v = self.vector_to_matrix(v, self.input_dim)
            if len(w.shape) < 2:
                w = self.vector_to_matrix(w, self.input_dim)
            if return_trajectory:
                return np.diag(v.T.dot(self.cost_data.input_product.dot(w)))
            else:
                return np.vdot(time_norm, np.diag(v.T.dot(self.cost_data.input_product.dot(w))))
        elif space_norm == 'identity':
            if len(v.shape) < 2:
                v = self.vector_to_matrix(v, self.output_dim)
            if len(w.shape) < 2:
                w = self.vector_to_matrix(w, self.output_dim)
            if return_trajectory:
                return np.diag(v.T.dot(w))
            else:
                return np.vdot(time_norm, np.diag(v.T.dot(w)))

    def space_time_norm(self, v, space_norm='L2', time_norm=None):
        return np.sqrt(self.space_time_product(v, v, space_norm, time_norm=time_norm))

    def rel_error_norm(self, U1, U2, space_norm='L2'):
        return self.space_time_norm(U1 - U2, space_norm=space_norm) / self.space_time_norm(U1, space_norm=space_norm)

    # %% optimization algorithms

    def set_default_options(self, tol=1e-8, maxit=200, save=False, plot=True, print_info=True, solve_true=False):
        options = {'print_info': print_info,
                   'print_final': True,
                   'plot': plot,
                   'save': save,
                   'path': None,
                   'tol': tol,
                   'maxit': maxit,
                   'solve_true': solve_true,
                   }
        return options

    def solve_ocp(self, U_0, options=None):
        if options is None:
            options = self.set_default_options()
        u_opt, history = self.solve_BB(U_0.flatten(), options)
        return u_opt, history

    def solve_BB(self, u_0, options, ):
        if options['print_info']:
            print('#############################################################')
        if options['print_info']:
            print("Starting BB ")
        # use optimize the discretize
        BBnorm = lambda x: self.space_time_norm(x, 'control')
        BBproduct = lambda x, y: self.space_time_product(x, y, 'control')
        BBgrad = lambda x: self.gradJ_OBD(x)

        # initialize
        t = time()
        k = 0
        u_km1 = u_0
        grad_km1, out, Y, P = BBgrad(u_km1)
        grad_norm_km1 = BBnorm(grad_km1)
        if grad_norm_km1 <= options['tol']:
            iter_flag = False
            u_k = u_km1
            history = {'grad_norm': [grad_norm_km1],
                       'u_list': [u_k]
                       }

            grad_norm0 = grad_norm_km1
            grad_norm = grad_norm0
            grad_k = grad_km1
            s = 0
        else:
            iter_flag = True
            s = 1
            u_k = u_km1 - s * grad_km1
            grad_k, out, Y, P = BBgrad(u_k)
            grad_norm0 = BBnorm(grad_km1)
            grad_norm = grad_norm0
            history = {'grad_norm': [grad_norm_km1, grad_norm],
                       'time_stages': [],
                       'u_list': [u_k]
                       }

        if options['print_info']:
            print(f"k: {-1:3}, grad_norm = {grad_norm_km1: 2.6e}")
            print(
                f"k: {k:3}, grad_norm = {grad_norm: 2.6e}, rel grad_norm = {grad_norm / grad_norm0: 2.6e}, alpha: {s: 2.6e}")

        # BB loop
        while grad_norm > max(options['tol'], options['tol'] * grad_norm0) and k < options['maxit'] and iter_flag:

            # compute BB steplength
            sk = u_k - u_km1
            dk = grad_k - grad_km1
            skdk = BBproduct(sk, dk)
            if k % 2 == 0:
                alpha_k = BBproduct(dk, dk) / skdk
            else:
                alpha_k = skdk / BBproduct(sk, sk)

            # update
            u_km1 = u_k
            u_k = u_k - grad_k / alpha_k
            grad_km1 = grad_k

            # compute new gradient and its norm
            grad_k, out, Y, P = BBgrad(u_k)
            grad_norm = BBnorm(grad_k)

            # update history
            history['grad_norm'].append(grad_norm)
            history['u_list'].append(u_k)
            k += 1
            if options['print_info']:
                print(
                    f"k: {k:3}, grad_norm = {grad_norm: 2.6e}, rel grad_norm = {grad_norm / grad_norm0: 2.6e}, alpha: {1 / alpha_k: 2.6e}")

        # finalize
        U_opt = self.vector_to_matrix(u_k, self.input_dim)
        history['U_opt'] = U_opt
        history['Y_opt'] = Y
        history['P_opt'] = P
        history['out_opt'] = out
        history['time'] = time() - t
        history['k'] = k
        if k == options['maxit']:
            history[
                'flag'] = f'BB reached maxit of k = {k:2} iterations in {history["time"]: .3f} seconds with gradient norm of {grad_norm:2.6e}, rel grad_norm = {grad_norm / grad_norm0: 2.6e}.'
        else:
            history[
                'flag'] = f'BB converged in k = {k:2} iterations in {history["time"]: .3f} seconds with gradient norm of {grad_norm:2.6e}, rel grad_norm = {grad_norm / grad_norm0: 2.6e}.'
        if options['print_final']:
            print(history['flag'])
        if options['print_info']:
            print('#############################################################')
        if options['plot']:
            plt.figure()
            plt.semilogy(history['grad_norm'])
            plt.title(r'BB convergence of $\|\nabla F(u_k)\|_U$')
            plt.xlabel(r'$k$')
            if options['save']:
                plt.savefig(options['path'])
        return U_opt, history

    # %% error estimation

    def optimal_control_est(self, Ur, Y=None, P=None):

        # if control is reduced reconstruct optimal control
        if Y is not None and P is not None:
            dJ, output, Y, P = self.gradJ_OBD(Ur, Y=Y, P=P)
            grad_norm = self.space_time_norm(dJ, 'control')
            est = grad_norm / self.cost_data.weights[1]
            return est, Y, P, None, dJ, grad_norm

        # solve forward and backward and evaluate gard norm
        dJ, output, Y, P = self.gradJ_OBD(Ur)
        grad_norm = self.space_time_norm(dJ, 'control')
        est = grad_norm / self.cost_data.weights[1]
        return est, Y, P, output, dJ, grad_norm

    # %% plot

    def visualize_1d(self, output, title=None, semi=False, time=None):
        plt.figure()
        if time is None:
            timeint = self.time_disc.t_v
        else:
            timeint = self.time_disc.t_v[:time]

        if title is not None:
            plt.title(title)

        if semi:
            plt.semilogy(timeint, output)
        else:
            plt.plot(timeint, output)
        plt.xlabel(r'$t$')
        plt.show()

    def visualize_1d_many(self, outputs, strings, title=None, semi=False, time=None, markersize=4, path=None):
        assert len(outputs) == len(strings), 'this has to be the same'
        plt.figure()
        if time is None:
            timeint = self.time_disc.t_v
        else:
            timeint = self.time_disc.t_v[:time]

        if title is not None:
            plt.title(title)

        markers = ['o', 's', 'D', '^', 'v', 'x', '*']
        linestyles = ['-', '--', '-.', ':']

        for i in range(len(outputs)):
            if semi:
                plt.semilogy(timeint, outputs[i], label=strings[i],
                             linestyle=linestyles[i % len(linestyles)],
                             marker=markers[i % len(markers)], markersize=markersize, markevery=(i, 4)
                             )
            else:
                plt.plot(timeint, outputs[i], label=strings[i],
                         linestyle=linestyles[i % len(linestyles)],
                         marker=markers[i % len(markers)], markersize=markersize, markevery=(i, 4))

        plt.legend()
        plt.xlabel(r'$t$')
        if path is not None:
            plt.savefig(path, format="eps", bbox_inches="tight")
        plt.show()

    def fenics_plot_solution(self, y, title=''):
        yf = fenics.Function(self.space_disc.V)
        yf.vector()[:] = y
        plt.figure()
        c = fenics.plot(yf, title=title, mode='color')
        plt.colorbar(c)

    def plot_3d(self, y, title=None, path=None, elev=35, azim=-50, fontsize=20):

        # read model
        Nx = self.space_disc.Nx
        Ny = self.space_disc.Ny
        mesh = self.space_disc.mesh
        V = self.space_disc.V

        # get plot data
        dims = (Ny + 1, Nx + 1)
        X = np.reshape(mesh.coordinates()[:, 0], dims)
        Y = np.reshape(mesh.coordinates()[:, 1], dims)
        Z = np.reshape(y[fenics.vertex_to_dof_map(V)], dims)

        # plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection='3d')
        if title is not None:
            ax.set_title(title, fontsize=fontsize)
        ax.plot_surface(X, Y, Z, cmap='coolwarm')
        ax.view_init(elev=elev, azim=azim, roll=0)
        ax.set_xlabel(r"$x_1$", labelpad=14, fontsize=fontsize)
        ax.set_ylabel(r"$x_2$", labelpad=14, fontsize=fontsize)
        ax.set_zlabel("", labelpad=18, fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.tick_params(axis='z', labelsize=fontsize)

        if path is not None:
            plt.savefig(path, format="eps", bbox_inches="tight")
        # plt.figure()
        plt.show()

    # %% helpers

    def matrix_to_vector(self, V):
        return V.flatten()

    def vector_to_matrix(self, v, dim):
        return v.reshape(dim, self.time_disc.K)

    def derivative_check(self, mode=1):
        print('Derivative check running ...')
        f = self.J
        df = self.gradJ_OBD
        Eps = np.array([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        u = np.random.random((self.input_dim, self.time_disc.K))
        du = np.random.random((self.input_dim, self.time_disc.K))
        T = np.zeros(np.shape(Eps))
        T2 = T
        ff = f(u)

        # Compute central and right-side difference quotient
        for i in range(len(Eps)):
            # print(Eps[i])
            f_plus = f(u + Eps[i] * du)
            f_minus = f(u - Eps[i] * du)
            if mode == 1:
                ddd = self.space_time_product(df(u)[0], du, 'control')
                T[i] = abs(((f_plus - f_minus) / (2 * Eps[i])) - ddd)
                T2[i] = abs(((f_plus - ff) / (Eps[i])) - ddd)
            else:
                T[i] = abs(((f_plus - f_minus) / (2 * Eps[i])) - df(u, du))
                T2[i] = abs(((f_plus - ff) / (Eps[i])) - df(u, du))

        # Plot
        plt.figure()
        plt.xlabel('$eps$')
        plt.ylabel('$J$')
        plt.loglog(Eps, Eps, label='O(eps)')
        plt.loglog(Eps, T2, 'ro--', label='Test')
        plt.legend(loc='upper left')
        plt.grid()
        plt.title("Rightside difference quotient")
        print('Derivative check finished ...')
