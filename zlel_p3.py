# -*- coding: utf-8 -*-
"""
.. module:: zlel_p3.py
:synopsis: Module for the 3rd lab of LNLC subject.
Contains the device physics functions to calculate the Newton-Raphson discrete
equivalents for non-linear components (Diodes and BJTs).   
.. moduleauthor:: Aitor Serrano Murua (aserrano052@ikasle.ehu.eus), Erik 
Elorduy Bravo (eelorduy005@ikasle.ehu.eus)
"""

import numpy as np
import sys

if __name__ == "zlel.zlel_p3":
    import zlel.zlel_p1 as zl1
    import zlel.zlel_p2 as zl2
else:
    import zlel_p1 as zl1
    import zlel_p2 as zl2

# Thermal voltage at T = 300K: VT = k*T / q
VT = 8.6173324e-5 * 300.0


def diode_NR(I0, nD, Vdj):
    """
    Calculates the conductance (g) and independent current (I) of a diode for 
    its Newton-Raphson discrete equivalent.

    Given the diode equation:
        Id = I0 * (e^(Vd / n*VT) - 1)

    The NR discrete equivalent is:
        i_{j+1} + g * v_{j+1} = I

    Parameters
    ----------
    I0 : float
        Saturation current of the diode.
    nD : float
        Ideality factor of the diode.
    Vdj : floar
        Diode voltage at the current iteration.

    Returns
    -------
    gd : float
        Conductance of the NR discrete equivalent (g).
    Id : float
        Independent current source of the NR discrete equivalent (I).

    """

    # 1. Calculate the exponent
    exponent = Vdj / (nD * VT)

    # 2. Prevent math overflow
    exponent = min(exponent, 100.0)

    e_val = np.exp(exponent)

    # 3. Calculate g and I
    gd = -(I0 / (nD * VT)) * e_val
    Id = I0 * (e_val - 1.0) + (gd * Vdj)

    return gd, Id


def bjt_NR(Ies, Ics, betaF, Vbej, Vbcj):
    """
    Calculates the 2x2 conductance matrix parameters and current vectors for a
    BJT NPN transistor using Ebers-Moll equations.

    The NR discrete equivalent equations are:
        iE + g11*VBE + g12*VBC = IE
        iC + g21*VBE + g22*VBC = IC

    Parameters
    ----------
    Ies : float
        Emitter saturation current.
    Ics : float
        Collector saturation current.
    betaF : float
        Forward common-emitter current gain.
    Vbej : float
        Base-Emitter voltage at current iteration (j).
    Vbcj : float
        Base-Collector voltage at current iteration (j).

    Returns
    -------
    g11 : float
        Conductance matrix value 11.
    g12 : float
        Conductance matrix value 12.
    g21 : float
        Conductance matrix value 21.
    g22 : float
        Conductance matrix value 22.
    IE : float
        Emitter independent current source value.
    IC : float
        Collector independent current source value.

    """
    # 1. Calculate the Alphas
    alphaF = betaF / (1.0 + betaF)
    alphaR = (Ies / Ics) * alphaF

    # 2. Prevent overflow on the exponents
    exp_be = min(Vbej / VT, 100.0)
    exp_bc = min(Vbcj / VT, 100.0)

    e_be = np.exp(exp_be)
    e_bc = np.exp(exp_bc)

    # 3. Calculate the conductance matrix (g)
    g11 = -(Ies / VT) * e_be
    g22 = -(Ics / VT) * e_bc
    g12 = -alphaR * g22
    g21 = -alphaF * g11

    # 4. Calculate the independent currents (I)
    IE = (g11 * Vbej) + (g12 * Vbcj) + Ies * (e_be - 1.0) - (alphaR * Ics *
                                                             (e_bc - 1.0))
    IC = (g21 * Vbej) + (g22 * Vbcj) - (alphaF * Ies *
                                        (e_be - 1.0)) + Ics * (e_bc - 1.0)

    return g11, g12, g21, g22, IE, IC


def solve_nl_circuit(br_el, br_val, br_ctr, b, n, A, t=0.0, is_op=False):
    """
    Solves a non-linear circuit using the Newton-Raphson method.

    Parameters
    ----------
    br_el : numpy.ndarray
        Array of strings with the expanded branch names. Size: (b,).
    br_val : numpy.ndarray
        Array of floats containing the values for each branch. Size: (b, 3).
    br_ctr : numpy.ndarray
        Array of strings with the control element names for each branch. Size: 
        (b,).
    b : int
        Total number of branches in the circuit
    n : int
        Total number of nodes in the circuit.
    A : numpy.ndarray
        Reduced incidence matrix.
    t : float, optional
        Time (for transient analysis). The default is 0.0.
    is_op : boolean, optional
        Flag for .OP analysis. The default is False.

    Returns
    -------
    sol : numpy.ndarray
        The converged Tableau solution.

    """
    # 1. Newton-Raphson Parameters
    N_max = 100
    epsilon = 1e-5

    # 2. Initialize branch voltages (v_j)
    v_j = np.zeros(b, dtype=float)
    for i in range(b):
        type_letter = br_el[i][0].upper()
        if type_letter in ['D', 'Q']:
            # Non-linear elements start at 0.6V
            v_j[i] = 0.6

    # 3. The Newton-Raphson Iteration Loop
    for iter_count in range(N_max):
        # a) Build Physics matrices using current v_j guesses
        M, N_mat, Us = zl1.build_bce(
            br_el, br_val, br_ctr, b, t=t, is_op=is_op, v_j=v_j)

        # b) Build and solve the linear Tableau system
        T, U = zl2.build_tableau(A, M, N_mat, Us, b, n)

        try:
            sol = np.linalg.solve(T, U)
        except np.linalg.LinAlgError:
            sys.exit("Error solving Tableau equation! Singular Matrix "
                     "in NR loop.")

        # c) Extract the new branch voltages (v_j_new) from the solution vector
        # The solution vector structure is:
        # [e_1...e_n-1, v_1...v_b, i_1...i_b]^T
        # Therefore, branch voltages start at index (n-1) and end at (n-1 + b)
        v_j_new = sol[n - 1: n - 1 + b, 0]

        # d) Check for convergence
        # Calculate the absolute difference between new and old voltages
        max_diff = np.max(np.abs(v_j_new - v_j))

        if max_diff < epsilon:
            # Convergence reached
            return sol

        # e) Update v_j for the next iteration
        v_j = np.copy(v_j_new)

    # If the loop finishes without returning, convergence failed
    sys.exit(f"ERROR: Newton-Raphson did not converge after {N_max} "
             "iterations!")


if __name__ == "__main__":
  pass
