# -*- coding: utf-8 -*-
"""
.. module:: zlel_main.py
    :synopsis:

.. moduleauthor:: Aitor Serrano Murua (aserrano052@ikasle.ehu.eus), Erik 
Elorduy Bravo (eelorduy005@ikasle.ehu.eus)


"""
import sys
import zlel.zlel_p1 as zl1

if __name__ == "__main__":

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "cirs/Adibidea1.cir"

    print("==========================================")
    print(f"  ZLEL Circuit Solver - Analyzing: {filename} ")
    print("==========================================\n")

    # Parse the circuit
    [cir_el, cir_nd, cir_val, cir_ctr] = zl1.cir_parser(filename)

    # Expand to branches
    [br_el, br_nd, b] = zl1.build_branches(cir_el, cir_nd)

    # Get nodes and elements
    nodes = zl1.get_nodes(cir_nd)
    el_num = len(cir_el)
    n = len(nodes)

    # Print the formated information
    zl1.print_cir_info(br_el, br_nd, b, n, nodes, el_num)

    # Print incidence matrices
    [Aa, A] = zl1.build_incidence_matrix(br_nd, b, nodes, n)

    # Check for errors
    zl1.check_circuit_errors(Aa, br_el, cir_el, cir_val, nodes)

    print("Incidence matrix:")
    print(Aa)

"""
https://stackoverflow.com/questions/419163/what-does-if-name-main-do
https://stackoverflow.com/questions/19747371/
python-exit-commands-why-so-many-and-when-should-each-be-used
"""
