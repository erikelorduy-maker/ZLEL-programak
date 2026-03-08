# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis:

.. moduleauthor:: Aitor Serrano Murua (aserrano052@ikasle.ehu.eus), Erik 
Elorduy Bravo (eelorduy005@ikasle.ehu.eus)


"""
import sys
import numpy as np
import zlel.zlel_p1 as zl1
import zlel.zlel_p2 as zl2


if __name__ == "__main__":

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "cirs/1_zlel_Rak.cir"

    # Parse the circuit
    [cir_el, cir_nd, cir_val, cir_ctr, cir_sim] = zl1.cir_parser(filename)

    # Expand to branches
    [br_el, br_nd, br_val, br_ctr, b] = zl1.build_branches(cir_el, cir_nd,
                                                           cir_val, cir_ctr)

    # Get nodes and elements
    nodes = zl1.get_nodes(cir_nd)
    el_num = len(cir_el)
    n = len(nodes)

    # Build incidence matrices
    [Aa, A] = zl1.build_incidence_matrix(br_nd, b, nodes, n)

    # Check for errors
    zl1.check_circuit_errors(Aa, br_el, cir_el, cir_val, nodes)

    # --- .PR Analysis (Print Circuit Info) ---
    if 'pr' in cir_sim:
        zl1.print_cir_info(cir_el, cir_nd, b, n, nodes, el_num)
        print("\nIncidence Matrix: ")
        print(Aa)

    # --- .OP Analysis (Operating Point) ---
    if 'op' in cir_sim:
        # 1. Build the BCE physics matrices (M, N, Us). We use t=0.0 for DC.
        M, N, Us = zl1.build_bce(br_el, br_val, br_ctr, b, t=0.0, is_op=True)

        # 2. Assemble the massive Tableau system (T * x = U)
        T, U = zl2.build_tableau(A, M, N, Us, b, n)

        # 3. Solve the linear system instantly
        try:
            solution = np.linalg.solve(T, U)
        except np.linalg.LinAlgError:
            sys.exit("Error solving Tableau equations, check if det(T) != 0.")

        # 4. Print the formatted solution using the professor's template
        zl2.print_solution(solution, b, n)

    # --- .DC Analysis (DC Sweep) ---
    if 'dc' in cir_sim:
        zl2.simulate_dc(cir_sim, filename, br_el, br_val, br_ctr, b, n, A)

    # --- .TR Analysis (Transient Sweep) ---
    if 'tr' in cir_sim:
        zl2.simulate_tr(cir_sim, filename, br_el, br_val, br_ctr, b, n, A)


"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
https://stackoverflow.com/questions/19747371/
python-exit-commands-why-so-many-and-when-should-each-be-used
"""

