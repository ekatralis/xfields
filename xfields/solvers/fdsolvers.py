# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import numpy as np
from scipy.constants import epsilon_0
from numpy import pi

from .base import Solver

from xobjects import context_default
import xtrack as xt
import xobjects as xo
# from PyPIC.geom_impact_poly import polyg_cham_geom_object as PyPIC_Chamber
from ._temp.geom_impact_poly import polyg_cham_geom_object as PyPIC_Chamber
import scipy.sparse as scsp

from tqdm import tqdm


class FDStaircaseSolver2p5D(Solver):

    def __init__(self, chamber: PyPIC_Chamber, 
                 x_grid: np.ndarray,
                 y_grid: np.ndarray,
                 z_grid: np.ndarray = None,
                 sparse_solver: str = None, 
                 sparse_solver_kwargs: dict = None,
                 remove_external_nodes_from_mat: bool = True,
                 context: xo.context.XContext = None
                 ):
        
        if context is None:
            context = context_default

        self.context = context
        print("[XFields] Staircase init")
        # To generate pairs of xy coordinates. We iterate with F contiguity in mind
        # as such, we use the "indexing='ij'" option for meshgrid
        nx = len(x_grid)
        ny = len(y_grid)
        if z_grid is not None:
            nbatches = len(z_grid)
        else:
            nbatches = 0

        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        
        assert dx == dy, ("Only uniform grids can be used with this solver")
        Dh = dx

        [xn, yn] = np.meshgrid(x_grid,y_grid, indexing = 'ij')
        xn = xn.flatten(order='F')
        yn = yn.flatten(order='F')
        
        # import matplotlib.pyplot as plt
        flag_outside_n=chamber.is_outside(xn,yn)
        flag_inside_n=~(flag_outside_n)
        # plt.figure(12)
        # plt.scatter(xn[flag_inside_n],yn[flag_inside_n],color = 'b')
        # plt.scatter(xn[flag_outside_n],yn[flag_outside_n],color = 'r')
        # plt.show()

        A=scsp.lil_matrix((nx*ny,nx*ny))

        # Build A matrix
        for u in tqdm(range(0,nx*ny)):
            if flag_inside_n[u]:
                A[u,u] = -(4./(Dh*Dh))
                A[u,u-1]=1./(Dh*Dh);     #phi(i-1,j)
                A[u,u+1]=1./(Dh*Dh);     #phi(i+1,j)
                A[u,u-nx]=1./(Dh*Dh);    #phi(i,j-1)
                A[u,u+nx]=1./(Dh*Dh);    #phi(i,j+1)
            else:
                # external nodes
                A[u,u]=1.

        A=A.tocsr()

        # plt.figure(13)
        # plt.spy(A, markersize=0.2) 
        # A_coo = A.tocoo()

        # # plt.figure(figsize=(6, 6))
        # plt.scatter(A_coo.col, A_coo.row, s=1, c=A_coo.data)
        # plt.gca().invert_yaxis()
        # plt.xlabel("Column index")
        # plt.ylabel("Row index")
        # plt.title("Non-zero entries colored by value")
        # plt.colorbar(label="Value")
 
        # plt.show()

        if remove_external_nodes_from_mat:
            diagonal = A.diagonal()
            N_full = len(diagonal)
            indices_non_id = np.where(diagonal!=1.)[0]
            N_sel = len(indices_non_id)
            Msel = scsp.lil_matrix((N_full, N_sel))
            for ii, ind in enumerate(indices_non_id):
                Msel[ind, ii] =1.
        else:
            diagonal = A.diagonal()
            N_full = len(diagonal)
            Msel = scsp.lil_matrix((N_full, N_full))
            for ii in range(N_full):
                Msel[ii, ii] =1.
        Msel = Msel.tocsr()
        Asel = Msel.T*A*Msel
        
        self.sparse_lib = self.context.splike_lib.sparse
        self.Asel = self.sparse_lib.csr_matrix(Asel)
        self.Msel = self.sparse_lib.csr_matrix(Msel)
        self.MselT = self.sparse_lib.csr_matrix(Msel.T)

        if sparse_solver_kwargs is None:
            sparse_solver_kwargs = {}

        # self.solver = self.context.factorized_sparse_solver(self.Asel, 
        #                                       n_batches=nbatches, 
        #                                       force_solver=sparse_solver,
        #                                       solverKwargs = sparse_solver_kwargs
        #                                       )
        self.solver = xo.sparse.factorized_sparse_solver(self.Asel, 
                                              n_batches=nbatches, 
                                              force_solver=sparse_solver,
                                              solverKwargs = sparse_solver_kwargs,
                                              context = self.context
        )
        
        self.eps0_ctx = self.context.nparray_to_context_array(
            np.array(epsilon_0)
        )


    def solve(self,rho):
        if rho.ndim > 2:
            b = - rho.reshape(-1, rho.shape[-1], order = 'F') / self.eps0_ctx  # (nx*ny, nz)
        else:
            b = - rho.reshape(-1, order='F') / self.eps0_ctx  # (nx*ny)
            
        b = self.MselT@b
        phi_sel = self.solver.solve(b)
        phi = self.Msel@phi_sel
        
        return phi.reshape(rho.shape, order='F')
    

class FDShortleyWellerSolver2p5D(Solver):

    def __init__(self, chamber: PyPIC_Chamber, 
                 x_grid: np.ndarray,
                 y_grid: np.ndarray,
                 z_grid: np.ndarray = None,
                 sparse_solver: str = None, 
                 sparse_solver_kwargs: dict = None,
                 remove_external_nodes_from_mat: bool = True,
                 tol_stem: float = 0.01,
                 tol_der: float = 0.1,
                 context: xo.context.XContext = None
                 ):
        if context is None:
            context = context_default

        self.context = context
        print("[XFields] ShortleyWeller init")
        # To generate pairs of xy coordinates. We iterate with F contiguity in mind
        # as such, we use the "indexing='ij'" option for meshgrid
        nx = len(x_grid)
        ny = len(y_grid)
        if z_grid is not None:
            nbatches = len(z_grid)
        else:
            nbatches = 0

        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        
        assert dx == dy, ("Only uniform grids can be used with this solver")
        Dh = dx

        [xn, yn] = np.meshgrid(x_grid,y_grid, indexing = 'ij')
        xn = xn.flatten(order='F')
        yn = yn.flatten(order='F')
        
        # import matplotlib.pyplot as plt
        flag_outside_n = chamber.is_outside(xn,yn)
        flag_inside_n = ~flag_outside_n
        A=scsp.lil_matrix((nx*ny,nx*ny)); #allocate a sparse matrix
        Dx=scsp.lil_matrix((nx*ny,nx*ny)); #allocate a sparse matrix
        Dy=scsp.lil_matrix((nx*ny,nx*ny)); #allocate a sparse matrix

        list_internal_force_zero = []
        # Build A Dx Dy matrices 
        na = lambda x:np.array([x])
        for u in tqdm(range(0,nx*ny), desc="Mat Assembly"):
            if flag_inside_n[u]:

                #Compute Shortley-Weller coefficients
                if flag_inside_n[u-1]: #phi(i-1,j)
                    hw = Dh
                else:
                    x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamber.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u-1]), na(yn[u-1]), na(0.), resc_fac=.995, flag_robust=False)
                    hw = np.abs(x_int[0]-xn[u])

                if flag_inside_n[u+1]: #phi(i+1,j)
                    he = Dh
                else:
                    x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamber.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u+1]), na(yn[u+1]), na(0.), resc_fac=.995, flag_robust=False)
                    he = np.abs(x_int[0]-xn[u])

                if flag_inside_n[u-nx]: #phi(i,j-1)
                    hs = Dh
                else:
                    x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamber.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u-nx]), na(yn[u-nx]), na(0.), resc_fac=.995, flag_robust=False)
                    hs = np.abs(y_int[0]-yn[u])
                    #~ print hs

                if flag_inside_n[u+nx]: #phi(i,j+1)
                    hn = Dh
                else:
                    x_int,y_int,z_int,Nx_int,Ny_int, i_found_int = chamber.impact_point_and_normal(na(xn[u]), na(yn[u]), na(0.), na(xn[u+nx]), na(yn[u+nx]), na(0.), resc_fac=.995, flag_robust=False)
                    hn = np.abs(y_int[0]-yn[u])
                    #~ print hn


                # Build A matrix
                if hn<Dh*tol_stem or hs<Dh*tol_stem or hw<Dh*tol_stem or he<Dh*tol_stem: # nodes very close to the bounday
                    A[u,u] =1.
                    list_internal_force_zero.append(u)
                    #print u, xn[u], yn[u]
                else:
                    A[u,u] = -(2./(he*hw)+2/(hs*hn))
                    A[u,u-1]=2./(hw*(hw+he));     #phi(i-1,j)nx
                    A[u,u+1]=2./(he*(hw+he));     #phi(i+1,j)
                    A[u,u-nx]=2./(hs*(hs+hn));    #phi(i,j-1)
                    A[u,u+nx]=2./(hn*(hs+hn));    #phi(i,j+1)

                # Build Dx matrix
                if hn<Dh*tol_der:
                    if hs>=Dh*tol_der:
                        Dy[u,u] = -1./hs
                        Dy[u,u-nx]=1./hs
                elif hs<Dh*tol_der:
                    if hn>=Dh*tol_der:
                        Dy[u,u] = 1./hn
                        Dy[u,u+nx]=-1./hn
                else:
                    Dy[u,u] = (1./(2*hn)-1./(2*hs))
                    Dy[u,u-nx]=1./(2*hs)
                    Dy[u,u+nx]=-1./(2*hn)


                # Build Dy matrix	
                if he<Dh*tol_der:
                    if hw>=Dh*tol_der:
                        Dx[u,u] = -1./hw
                        Dx[u,u-1]=1./hw
                elif hw<Dh*tol_der:
                    if he>=Dh*tol_der:
                        Dx[u,u] = 1./he
                        Dx[u,u+1]=-1./(he)
                else:
                    Dx[u,u] = (1./(2*he)-1./(2*hw))
                    Dx[u,u-1]=1./(2*hw)
                    Dx[u,u+1]=-1./(2*he)
            else:
                # external nodes
                A[u,u]=1.
        if remove_external_nodes_from_mat:
            diagonal = A.diagonal()
            N_full = len(diagonal)
            indices_non_id = np.where(diagonal!=1.)[0]
            N_sel = len(indices_non_id)
            Msel = scsp.lil_matrix((N_full, N_sel))
            for ii, ind in enumerate(indices_non_id):
                Msel[ind, ii] =1.
        else:
            diagonal = A.diagonal()
            N_full = len(diagonal)
            Msel = scsp.lil_matrix((N_full, N_full))
            for ii in range(N_full):
                Msel[ii, ii] =1.
        Msel = Msel.tocsr()
        Asel = Msel.T*A*Msel
        
        self.sparse_lib = self.context.splike_lib.sparse
        self.Asel = self.sparse_lib.csr_matrix(Asel)
        self.Msel = self.sparse_lib.csr_matrix(Msel)
        self.MselT = self.sparse_lib.csr_matrix(Msel.T)
        self.Dx = self.sparse_lib.csr_matrix(Dx)
        self.Dy = self.sparse_lib.csr_matrix(Dy)

        if sparse_solver_kwargs is None:
            sparse_solver_kwargs = {}

        # self.solver = self.context.factorized_sparse_solver(self.Asel, 
        #                                       n_batches=nbatches, 
        #                                       force_solver=sparse_solver,
        #                                       solverKwargs = sparse_solver_kwargs
        #                                       )
        self.solver = xo.sparse.factorized_sparse_solver(self.Asel, 
                                              n_batches=nbatches, 
                                              force_solver=sparse_solver,
                                              solverKwargs = sparse_solver_kwargs,
                                              context = self.context
        )
        
        self.eps0_ctx = self.context.nparray_to_context_array(
            np.array(epsilon_0)
        )

    def solve(self,rho):
        if rho.ndim > 2:
            b = - rho.reshape(-1, rho.shape[-1], order = 'F') / self.eps0_ctx  # (nx*ny, nz)
        else:
            b = - rho.reshape(-1, order='F') / self.eps0_ctx  # (nx*ny)
            
        b = self.MselT@b
        phi_sel = self.solver.solve(b)
        phi = self.Msel@phi_sel
        
        return phi.reshape(rho.shape, order='F')