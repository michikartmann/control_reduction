#######################################################################################################################
#######################################################################################################################
# This file is part of the paper "Optimality-Based Control Space Reduction for Infinite-Dimensional Control Spaces"
# by Michael Kartmann, Stefan Volkwein
## Author of this code: Michael Kartmann
#######################################################################################################################
#######################################################################################################################

import scipy.sparse as sps
from scipy import linalg
import numpy as np
from model import model
from methods import Collection
import matplotlib.pyplot as plt
from time import perf_counter


def project_model(model, U, V=None, product=None, project_control=False):
    '''
    

    Parameters
    ----------
    model_ : TYPE
        DESCRIPTION.
    U : r x N
        Left projection matrix
    V : r x N, optional
        Right Projection matrix

    Returns
    -------
    projected_pde : TYPE
        DESCRIPTION.

    '''

    # Attention: this code is for Galerkin projection and has to be modified for Petrov Galerkin projections

    # init
    pde, cost = model.pde, model.cost_data
    if V is None:  # then do Galerkin projection
        V = U

    # pde
    projected_pde = Collection()
    if model.isSwitchModel():
        projected_pde.type = 'SwitchROM'
        projected_pde.sigma = pde.sigma
        if 'StateDep' in model.pde.type:
            projected_pde.type += 'StateDep'
        projected_pde.error_est_constants = pde.error_est_constants

    elif model.isTimeVaryingModel():
        projected_pde.type = 'TimeVaryingROM'
        projected_pde.A_time_coefficient = pde.A_time_coefficient
        projected_pde.M_time_coefficient = pde.M_time_coefficient
        projected_pde.B_time_coefficient = pde.B_time_coefficient
        projected_pde.C_time_coefficient = pde.C_time_coefficient

    if model.isContinuousControl():
        projected_pde.type += 'continuousControl'

    if not project_control:
        projected_pde.type += '_not_reduced'

    projected_pde.A = []
    for AA in pde.A:
        projected_pde.A.append(U.T @ (AA.dot(V)))
    projected_pde.M = []
    for MM in pde.M:
        projected_pde.M.append(U.T @ (MM.dot(V)))
    projected_pde.B = []

    for BB in pde.B:
        if model.isContinuousControl() and project_control:
            projected_pde.B.append(U.T @ BB @ V)
        else:
            projected_pde.B.append(U.T @ BB)

    projected_pde.C = []
    for CC in pde.C:
        if CC.shape[0] == CC.shape[1] == pde.state_dim:
            assert CC.trace() == pde.state_dim, 'C ist not identity and it has to be projected ...'
            CC_proj = np.eye(np.shape(projected_pde.A[0])[0])
            projected_pde.C.append(CC_proj)

        else:
            projected_pde.C.append(CC @ V)

    # products
    projected_pde.products = {}
    for PP in pde.products.keys():
        MAT = pde.products[PP]
        projected_pde.products[PP] = (V.T @ (MAT.dot(V)))
    projected_pde.F = U.T @ (pde.F)

    if product is not None:
        projected_pde.y0 = U.T @ product @ (pde.y0)
    else:
        projected_pde.y0 = U.T @ (pde.y0)

    projected_pde.state_dim = np.shape(projected_pde.A[0])[0]
    projected_pde.input_dim = pde.input_dim

    # project cost
    projected_cost = Collection()
    projected_cost.weights = cost.weights

    # project control if desired?
    if cost.Ud.shape[0] == pde.state_dim and project_control:
        projected_cost.Mc_Ud = U.T @ cost.Mc_Ud
        projected_cost.input_product = U.T @ (cost.input_product.dot(U))
        projected_cost.Ud_Mc_Ud = cost.Ud_Mc_Ud
        projected_cost.Ud = U.T @ cost.Mc_Ud
        projected_pde.input_dim = np.shape(projected_pde.B[0])[0]
    else:
        projected_cost.Ud = cost.Ud
        projected_cost.Mc_Ud = cost.Mc_Ud
        projected_cost.input_product = cost.input_product
        projected_cost.Ud_Mc_Ud = cost.Ud_Mc_Ud
        projected_pde.input_dim = pde.input_dim

    if cost.Yd.shape[0] == pde.state_dim:
        projected_cost.Yd = None
        projected_cost.YT = None
        projected_cost.Mc_Yd = V.T @ cost.Mc_Yd
        projected_cost.Mc_YT = V.T @ cost.Mc_YT
        projected_cost.output_product = V.T @ (cost.output_product.dot(V))
        projected_cost.Yd_Mc_Yd = cost.Yd_Mc_Yd
        projected_cost.YT_Mc_YT = cost.YT_Mc_YT
        projected_pde.output_dim = np.shape(projected_pde.C[0])[0]

    else:
        projected_cost.Yd = cost.Yd
        projected_cost.YT = cost.YT
        projected_cost.Mc_Yd = cost.Mc_Yd
        projected_cost.Mc_YT = cost.Mc_YT
        projected_cost.output_product = cost.output_product
        projected_cost.Yd_Mc_Yd = cost.Yd_Mc_Yd
        projected_cost.YT_Mc_YT = cost.YT_Mc_YT
        projected_pde.output_dim = pde.output_dim

    return projected_pde, projected_cost


# %% POD

class pod_reductor():

    def __init__(self, model, model_toproject=None, H_prod=None, space_product=None, project_control=False):
        self.space_product = space_product
        self.project_control = project_control
        self.update_type = 'redo_svd'
        self.incremental_tol = 1e-14
        self.truncation_tol = 1e-15
        self.total_energy = None
        self.model = model

        if space_product is not None:
            start_time = perf_counter()
            self.Wchol = linalg.cholesky(space_product.todense())
            end_time = perf_counter()
            # print(f'Cholesky {end_time - start_time}')
        else:
            self.Wchol = None

        if model_toproject is None:
            self.model_toproject = model
        else:
            self.model_toproject = model_toproject

        if H_prod is not None:
            self.H_prod = H_prod

        # track snapshots data
        self.Snapshots = []
        self.snapshot_energy = []
        self.Ds = []
        self.XisMax = []
        self.XisMin = []

    def ROMtoFOM(self, u):
        return self.V_right @ u

    def FOMtoROM(self, U):
        if self.projection_product is None:
            return self.U_left.T @ U
        else:
            return self.U_left.T @ self.projection_product @ U

    def check_orthogonality(self):
        print(self.POD_Basis.T @ self.space_product @ self.POD_Basis)

    def project_and_build_error_est(self, model_to_project=None):

        # project and update basis
        rom_pod = self.project(U=self.U_left, V=self.V_right, product=self.projection_product,
                               H_prod=self.space_product, model_to_project=model_to_project)
        rom_pod.reductor = self
        self.rom_pod = rom_pod

        return self.rom_pod

    def get_rom(self, l, Snapshots, space_product, time_product, PODmethod, plot=False, model_to_project=None,
                pod_basis=None, pod_values=None):

        # print('ROM POD constructing ...')
        start_time = perf_counter()

        if self.space_product is None:
            self.space_product = space_product

        if model_to_project is None:
            model_to_project = self.model_toproject

        if pod_basis is None or pod_values is None:

            # compute energy for snapshots
            self.snapshot_energy = []
            for s in Snapshots:
                self.snapshot_energy.append(self.model.space_time_product(s, s, time_norm=time_product.diagonal(),
                                                                          space_mat=self.space_product))
            self.total_energy = sum(self.snapshot_energy)

            # get pod basis
            pod_basis, pod_values = self.pod_basis(Y=Snapshots,
                                                   l=l,
                                                   W=self.space_product,
                                                   D=time_product,
                                                   flag=PODmethod)

        # collect stuff
        self.POD_Basis = pod_basis
        self.POD_values = pod_values
        self.Singular_values = np.sqrt(pod_values)
        self.U_left = pod_basis
        self.V_right = pod_basis
        self.projection_product = self.space_product

        # plot vals if desired
        if 0:
            self.plot_pod_values()

        # project the model onto pod basis
        rom_pod = self.project(U=self.U_left, V=self.V_right, product=self.projection_product,
                               H_prod=self.space_product, model_to_project=model_to_project)
        rom_pod.reductor = self
        self.rom_pod = rom_pod
        end_time = perf_counter()
        print(f'ROM constructed in {end_time - start_time}')

        return rom_pod

    def project(self, U, V=None, product=None, H_prod=None, model_to_project=None):

        if model_to_project is None:
            model_to_project = self.model_toproject

        projected_pde, projected_cost = project_model(model_to_project, U, V, product, project_control=self.project_control)
        rom = model(projected_pde, projected_cost, model_to_project.time_disc, model_to_project.space_disc,
                    model_to_project.options)
        rom.type = projected_pde.type + 'POD'
        return rom

    def pod_basis(self, Y, l, W=None, D=None, flag=0, energy_tolerance=None):
        """
        #     Compute POD basis

        #     Parameters
        #     ----------
        #     Y: list,
        #         list containing different snapshot matrices
        #     l: int,
        #         Length of the POD-basis.
        #     W: ndarray, shape (n_x,n_x)
        #         Gramian of the Hilbert space X, that containts the snapshots.
        #     D: list of/or ndarray of shape (n_t,n_t)
        #         Matrix containing the weights of the time discretization.
        #     flag: int
        #         parameter deciding which method to use for computing the POD-basis
        #         (if flag==0 svd, flag == 1 eig of YY', flag == 2 eig of Y'Y (snapshot method).

        #     Returns
        #     -------
        #     POD_Basis: ndarray, shape (n_x,l)
        #                 matrix containing the POD-basis vectors
        #     POD_Values: ndarray, shape (l,)
        #            vector containing the eigenvalues of Yhat (see below)

        """

        ### init
        # set truncation tol
        tol = self.truncation_tol
        truncate_normalized_POD_values = False
        use_energy_content = True
        if energy_tolerance is None:
            energy_tolerance = 1
        if W is None:
            pass  # set it to euclidian product
        if D is None:
            pass  # set it to euclidian
        # compute square root of the diagonal matrix D
        if type(D) == list and 0:
            Dsqrt = [d.sqrt() for d in D]
        else:
            Dsqrt = D.sqrt()

            # construct snapshot matrix out of list
        K = len(Y)
        Dsqrt = [Dsqrt] * K
        Dsqrt = sps.block_diag(Dsqrt)
        Y = np.concatenate(Y, axis=1)

        assert flag == 0, 'only SVD options in this implementation'
        ### compute POD basis

        # scale matrix
        Yhat = self.Wchol @ Y @ Dsqrt
        l_min = min(l, min(Yhat.shape) - 1)
        # print(f'Basissize dropped from {l} to {l_min} due to rank condition of snapshot matrix.')

        # perform svd
        # self.Yhat = Yhat
        # print(f'lmin {l_min}')
        U, S, V = sps.linalg.svds(Yhat, k=l_min)  # linalg.svd(Yhat, full_matrices=False)#

        # get pod values
        POD_values = S ** 2

        # sort from biggest to lowest
        U = np.fliplr(U)
        POD_values = np.flipud(POD_values)

        # truncate w.r.t. the normalized singular values
        if truncate_normalized_POD_values:
            normalized_values = POD_values / POD_values[0]
        else:
            normalized_values = POD_values
        print(f'Smallest singular value {normalized_values[-1]} and biggest {normalized_values[0]}.')
        indices = normalized_values > tol
        POD_values = POD_values[indices]
        U = U[:, indices]
        print(f'Basissize dropped from {l_min} to {U.shape[1]} due to truncation of small modes.')
        POD_Basis = linalg.solve_triangular(self.Wchol, U, lower=False)

        # cut basis based on energy
        if energy_tolerance is not None and use_energy_content:
            size_before = len(POD_values)
            l_energy = 1
            local_energy = sum(POD_values[:l_energy])
            while local_energy / self.total_energy <= energy_tolerance and l_energy < size_before:
                l_energy += 1
                local_energy = sum(POD_values[:l_energy])
            print(f'Basissize dropped due to energy criterion from {size_before} to {l_energy}.')
            POD_Basis = POD_Basis[:, :l_energy]
            POD_values = POD_values[:l_energy]

        return POD_Basis, POD_values

    def plot_pod_values(self):
        plt.figure()
        plt.title('POD Eigenvalues decay')
        plt.semilogy(self.POD_values)
        plt.show()
