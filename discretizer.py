#######################################################################################################################
#######################################################################################################################
# This file is part of the paper "Optimality-Based Control Space Reduction for Infinite-Dimensional Control Spaces"
# by Michael Kartmann, Stefan Volkwein
## Author of this code: Michael Kartmann
#######################################################################################################################
#######################################################################################################################

import fenics as fenics
import numpy as np
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import factorized
from methods import Collection
from model import model


def discretize_stability_problem(T=1, Nx=100, K=101, model_options=None):
    #### time discretization
    time_disc = Collection()
    time_disc.t0 = 0
    time_disc.T = T
    time_disc.K = K
    time_disc.dt = (time_disc.T - time_disc.t0) / (time_disc.K - 1)
    time_disc.t_v = np.linspace(time_disc.t0, time_disc.T, num=time_disc.K)
    time_disc.D = time_disc.dt * np.ones(time_disc.K)
    time_disc.D_diag = diags(time_disc.D)
    tmp = time_disc.dt * np.ones(time_disc.K)
    tmp[0] = 1
    time_disc.D_diag_1 = diags(tmp)

    #### space discretization
    Lx = 1
    Ly = 1
    Ny = Nx
    x1 = 0
    y1 = 0
    x2 = Lx
    y2 = Ly
    lower_left = fenics.Point(x1, y1)
    upper_right = fenics.Point(x2, y2)
    mesh = fenics.RectangleMesh(lower_left, upper_right, Nx, Ny)
    tol = 1e-14

    class BoundaryL(fenics.SubDomain):  # left
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[0], x1, tol)  # x1

    class BoundaryR(fenics.SubDomain):  # right
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[0], x2, tol)  # x2

    class BoundaryLow(fenics.SubDomain):  # low
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[1], y1, tol)  # y1

    class BoundaryUp(fenics.SubDomain):  # up
        def inside(self, x, on_boundary):
            return on_boundary and fenics.near(x[1], y2, tol)  # y2

    boundary_markers = fenics.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)
    bx0 = BoundaryL()
    bx1 = BoundaryR()
    by0 = BoundaryLow()
    by1 = BoundaryUp()
    bx0.mark(boundary_markers, 0)
    bx1.mark(boundary_markers, 1)
    by0.mark(boundary_markers, 2)
    by1.mark(boundary_markers, 3)

    # redefine boundary integration measure
    ds = fenics.Measure('ds', domain=mesh, subdomain_data=boundary_markers)

    #### variational problem
    V = fenics.FunctionSpace(mesh, 'P', 1)
    y = fenics.TrialFunction(V)
    v = fenics.TestFunction(V)

    # data
    gamma_out = fenics.Constant(1)
    y_out = fenics.Constant(0.0)
    f = fenics.Constant(0.0)
    y0 = fenics.Expression('0*2*sin(pi*x[0])*sin(pi*x[1])', degree=1)  # fenics.Constant(0.0)

    #### B control operator
    B = fenics.assemble(y * v * fenics.dx)
    B = csr_matrix(fenics.as_backend_type(B).mat().getValuesCSR()[::-1])
    control_dofs = B.shape[1]
    B_time_coefficient = lambda t: 1

    #### M
    M = fenics.assemble(y * v * fenics.dx)
    M = csr_matrix(fenics.as_backend_type(M).mat().getValuesCSR()[::-1])
    M_time_coefficient = lambda t: 1

    #### A  
    boundary_conditions = {0: {'Robin': (gamma_out, y_out, 'no_control')},  # x = 0, left
                           1: {'Robin': (gamma_out, y_out, 'no_control')},  # x = 1, right
                           2: {'Robin': (gamma_out, y_out, 'no_control')},  # y = 0, lower
                           3: {'Robin': (gamma_out, y_out, 'no_control')}}  # y = 1, upper
    integrals_R_a = []
    integrals_R_L = []
    for i in boundary_conditions:
        if 'Robin' in boundary_conditions[i]:
            gamma_, y_, string = boundary_conditions[i]['Robin']
            integrals_R_a.append(gamma_ * y * v * ds(i))
            integrals_R_L.append(y_ * v * ds(i))
        else:
            assert 0, 'implement'

    # get coefficients for A and A
    A_time_coefficient = [lambda t: 1,  # diff
                          lambda t: 0,  # adv
                          lambda t: 2 + np.sin(4 * np.pi * t)]  # reac
    a_diff = fenics.Constant(1)
    a_adv = fenics.Expression(("-0.01*(x[0]+x[1])", "(x[1]*x[0])/2"), degree=2)
    a_reac = fenics.Constant(1.0)
    A1_diff = fenics.assemble(a_diff * fenics.dot(fenics.nabla_grad(y),
                                                  fenics.nabla_grad(v)) * fenics.dx)
    A1_diff = csr_matrix(fenics.as_backend_type(A1_diff).mat().getValuesCSR()[::-1])
    A2_adv = fenics.assemble(fenics.dot(fenics.nabla_grad(y), a_adv) * v * fenics.dx)
    A2_adv = csr_matrix(fenics.as_backend_type(A2_adv).mat().getValuesCSR()[::-1])
    A3_reac = fenics.assemble(a_reac * y * v * fenics.dx)
    A3_reac = csr_matrix(fenics.as_backend_type(A3_reac).mat().getValuesCSR()[::-1])
    A = [A1_diff, A2_adv, A3_reac]

    # norm for the reduced oeprator CS
    CS_norm = 1

    ##### F, y0, yd
    L = fenics.assemble(f * v * fenics.dx + sum(integrals_R_L))
    F = L.get_local()
    F = np.tile(F, [time_disc.K, 1]).T
    assert np.linalg.norm(F, 2) < 1e-14, 'We need zero rhs, or modify the code below...'
    y0 = fenics.interpolate(y0, V).vector().get_local()
    yd_exp = fenics.Expression('sin(t*2*pi*x[0])*sin(t*2*pi*x[1])', degree=3, t=time_disc.t0)
    yd_int = []
    for k in range(len(time_disc.t_v)):
        yd_exp.t = time_disc.t_v[k]
        yd_int.append(fenics.interpolate(yd_exp, V).vector().get_local())
    yd_int = np.array(yd_int).T

    #### C
    C = identity(len(y0))
    C_time_coefficient = lambda t: 1

    #### products
    L2 = fenics.assemble(y * v * fenics.dx)
    H10 = fenics.assemble(fenics.dot(fenics.nabla_grad(y), fenics.nabla_grad(v)) * fenics.dx)
    L2 = csr_matrix(fenics.as_backend_type(L2).mat().getValuesCSR()[::-1])
    H10 = csr_matrix(fenics.as_backend_type(H10).mat().getValuesCSR()[::-1])
    H1 = H10 + L2

    #### collect pde
    pde = Collection()
    pde.A = A
    pde.A_time_coefficient = A_time_coefficient
    pde.M = [M]
    pde.M_time_coefficient = M_time_coefficient
    pde.F = F
    pde.B = [B]
    pde.B_time_coefficient = B_time_coefficient
    pde.C = [C]
    pde.C_time_coefficient = C_time_coefficient
    pde.y0 = y0
    pde.state_dim = len(y0)
    pde.input_dim = control_dofs
    pde.output_dim = np.shape(C)[0]
    pde.products = {'H1': H1, 'L2': L2, 'H10': H10}
    pde.type = 'TimeVaryingFOM_DistributedControl'
    pde.factorizedV1 = factorized(H1)
    pde.constants = {'CS**2': CS_norm}

    #### collect space disc
    space_disc = Collection()
    space_disc.V = V
    space_disc.mesh = mesh
    space_disc.Nx = Nx
    space_disc.Ny = Ny
    space_disc.DirichletBC = None

    #### collect cost data
    cost_data = Collection()
    cost_data.weights = [1, 1e-2, 0]
    cost_data.Yd = yd_int
    cost_data.Ud = None
    cost_data.YT = yd_int[:, -1]
    cost_data.output_product = L2
    cost_data.input_product = M

    #### create fom
    fom = model(pde, cost_data, time_disc, space_disc, model_options)
    return fom


def get_y0(V, fenics_expression):
    y0 = fenics.interpolate(fenics_expression, V).vector().get_local()
    return y0
