# -*- coding: utf-8 -*-
"""
.. module:: zlel_p1.py
:synopsis: Module for 1st lab of LNLC subject.
It contains the circuit parser, a branch builder, info printing
and some integrity checks.
.. moduleauthor:: Aitor Serrano Murua (aserrano052@ikasle.ehu.eus), Erik 
Elorduy Bravo (eelorduy005@ikasle.ehu.eus)
"""

import sys
import numpy as np


def cir_parser(filename):
    """
    This function takes a .cir test circuit and parses it to 4 matrices.

    Parameters
    ----------
    filename : str
        String with the name and path of the .cir file to parse.

    Returns
    -------
    cir_el : numpy.ndarray
        Array of strings with the element names. Size: (e,).
    cir_nd : numpy.ndarray
        Array of integers with the nodes of the circuit. Size: (e, 4).
    cir_val : numpy.ndarray
        Array of floats with the values of the elements. Size: (e, 3).
    cir_ctr : numpy.ndarray
        Array of strings with the control element names. Size: (e,).
    """
    try:
        data = np.genfromtxt(filename, dtype=str)

        if data.size == 0:
            print("ERROR: File is empty.")
            sys.exit(1)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] != 9:
            print("ERROR: Every line must have exactly 9 elements "
                  "(non whitespace).")
            sys.exit(1)

        cir_nd = data[:, 1:5].astype(int)
        if not np.any(cir_nd == 0):
            print("ERROR: The circuit does not have a reference node (0).")
            sys.exit(1)

        cir_el = data[:, 0]
        cir_val = data[:, 5:8].astype(float)
        cir_ctr = data[:, 8]

        return cir_el, cir_nd, cir_val, cir_ctr

    except OSError:
        print(f"ERROR: The file {filename} was not found.")
        sys.exit(1)


def get_nodes(cir_nd):
    """
    Scans the node matrix and returns a sorted array of unique nodes.

    Parameters
    ----------
    cir_nd : numpy.ndarray
        Array of integers containing the nodes of the circuit.

    Returns
    -------
    nodes : numpy.ndarray
        1D array of unique, sorted integers representing the circuit's nodes.
    """
    return np.unique(cir_nd)


def build_branches(cir_el, cir_nd):
    """
    Expands the elements into branches for the digraph.

    Parameters
    ----------
    cir_el : numpy.ndarray
        Array of strings with the element names. Size: (e,)
    cir_nd : TYPE
        Array of integers with the nodes of the circuit. Size: (e, 4)

    Returns
    -------
    br_el : numpy.ndarray
        Array of strings with the expanded branch names (e.g., 'Q_1_be').
    br_nd : numpy.ndarray
        Array of integers with the branch nodes (positive and negative 
        terminals). Size: (b, 2).
    b : int
        Total number of branches in the circuit.
    """
    # 1. Identify which elements are multi-branch using NumPy string matching
    q_mask = np.char.startswith(np.char.lower(cir_el), 'q')
    a_mask = np.char.startswith(np.char.lower(cir_el), 'a')

    # Elements that are NOT transistors or op-amps
    other_mask = ~(q_mask | a_mask)

    # 2. Calculate total branches (b)
    # True counts as 1, False as 0 in np.sum
    b = np.sum(other_mask) + (2 * np.sum(q_mask)) + (2 * np.sum(a_mask))

    # 3. Initialize fixed-size NumPy arrays (No Python lists!)
    # '<U50': a string of up to 50 characters (needed to append '_be', etc.)
    br_el = np.empty(b, dtype='<U50')
    br_nd = np.empty((b, 2), dtype=int)

    # 4. Fill the arrays based on the topology rules
    idx = 0
    for i in range(len(cir_el)):
        el = cir_el[i]
        nd = cir_nd[i]

        if q_mask[i]:
            # BJT rules: Nodes are defined as [Collector, Base, Emitter] ->
            # [nd[0], nd[1], nd[2]]
            br_el[idx] = el + "_be"
            br_nd[idx] = [nd[1], nd[2]]  # Base to Emitter
            idx += 1

            br_el[idx] = el + "_bc"
            br_nd[idx] = [nd[1], nd[0]]  # Base to Collector
            idx += 1

        elif a_mask[i]:
            # OpAmp rules: Nodes are defined as [N+, N-, N_out, N_ref] ->
            # [nd[0], nd[1], nd[2], nd[3]]
            br_el[idx] = el + "_in"
            br_nd[idx] = [nd[0], nd[1]]  # N+ to N-
            idx += 1

            br_el[idx] = el + "_ou"
            br_nd[idx] = [nd[2], nd[3]]  # N_out to N_ref
            idx += 1

        else:
            # Standard 2-terminal components
            br_el[idx] = el
            br_nd[idx] = [nd[0], nd[1]]  # Positive to Negative
            idx += 1

    return br_el, br_nd, b


def build_incidence_matrix(br_nd, b, nodes, n):
    """
    Builds the full (Aa) and reduced (A) incidence matrices.

    Parameters
    ----------
    br_nd : numpy.ndarray
        Array of integers with the branch nodes. Size: (b, 2).
    b : int
        Total number of branches in the circuit.
    nodes : numpy.ndarray
        1D array of unique nodes in the circuit.
    n : int
        Total number of unique nodes in the circuit.

    Returns
    -------
    Aa : numpy.ndarray
        Full incidence matrix of the circuit. Size: (n, b).
    A : numpy.ndarray
        Reduced incidence matrix of the circuit (reference node removed). Size:
        (n-1, b).
    """
    # 1. Initialize a matrix of pure zeros
    Aa = np.zeros((n, b), dtype=int)

    # 2. Find the row index for every positive and negative terminal instantly
    # searchsorted looks at our sorted 'nodes' array and tells us exactly
    # which row index each node number belongs to.
    idx_plus = np.searchsorted(nodes, br_nd[:, 0])
    idx_minus = np.searchsorted(nodes, br_nd[:, 1])

    # Create an array representing the columns (0 to b-1)
    col_idx = np.arange(b)

    # 3. At these specific row/columns pairs, place a 1
    Aa[idx_plus, col_idx] = 1

    # At these specific row/columns pairs, place a -1"
    Aa[idx_minus, col_idx] = -1

    # 4. Create the Reduced Incidence Matrix (A)
    # Since our 'nodes' array is sorted, the reference node '0' is ALWAYS at row 0.
    # We slice the matrix to keep everything from row 1 to the end.
    A = Aa[1:, :]

    return Aa, A


def check_circuit_errors(Aa, br_el, cir_el, cir_val, nodes):
    """
    Analyzes the Incidence Matrix (Aa) to find physical circuit errors.

    Parameters
    ----------
    Aa : numpy.ndarray
        Full incidence matriz of the circuit. Size: (n, b).
    br_el : numpy.ndarray
        Array of strings with the expanded branch names (e.g., 'Q_1_be').
    cir_el : numpy.ndarray
        Array of strings with the original element names. Size: (e,).
    cir_val : numpy.ndarray
        Array of floats with the values of the elements. Size: (e, 3)

    Returns
    -------
    None.

    """
    # 1. Do all branches connect exactly 2 nodes?
    if not np.sum(Aa) == 0:
        sys.exit("ERROR: Fatal Matrix Error. A branch does not connect exactly"
                 "two nodes.")

    # 2. Floating nodes
    connections_per_node = np.sum(np.abs(Aa), axis=1)
    if np.any(connections_per_node == 1):
        sys.exit(
            "ERROR: The ciruit has a node(s) with a single connection (floating nodes).")

    # 3. Parallel voltage sources
    # Find all branches starting with 'V' or 'E'
    v_mask = np.char.startswith(np.char.lower(
        br_el), 'v') | np.char.startswith(np.char.lower(br_el), 'e')
    # Gets the column numbers for the voltage sources
    v_indices = np.flatnonzero(v_mask)

    if len(v_indices) > 1:
        # Extract only the columns of Aa that are voltage sources
        Aa_v = Aa[:, v_indices]

        # Calculate the correlation matrix: V^T * V
        correlation_matrix = np.dot(Aa_v.T, Aa_v)

        # Set the diagonal to 0 (so a source isn't flagged as parallel with itself!)
        np.fill_diagonal(correlation_matrix, 0)

        # Find where the dot product is 2 (parallel) or -2 (anti-parallel)
        # np.argwhere gives us a list of [row, col] pairs where the value is 2 or -2
        pairs = np.argwhere(np.abs(correlation_matrix) == 2)

        # If any pairs were found, we have parallel sources
        if len(pairs) > 0:
            # FAIL-FAST: We only care about the very first pair we find!
            idx1, idx2 = pairs[0]
            # Map back to branch indices
            br1, br2 = v_indices[idx1], v_indices[idx2]

            # Safely grab the actual voltage values from cir_val
            val1 = cir_val[cir_el == br_el[br1]][0, 0]
            val2 = cir_val[cir_el == br_el[br2]][0, 0]

            # If they are connected backwards (-2), flip val2 to compare them fairly
            if correlation_matrix[idx1, idx2] == -2:
                val2 = -val2

            if val1 != val2:
                sys.exit("ERROR: The voltage sources are in parallel with "
                         f"different values. ({br_el[br1]} eta {br_el[br2]})")

    # 3. SERIES CURRENT SOURCES
    i_mask = np.char.startswith(np.char.lower(
        br_el), 'i') | np.char.startswith(np.char.lower(br_el), 'g')
    i_indices = np.flatnonzero(i_mask)

    for node_idx in range(Aa.shape[0]):
        # How many current sources touch this node?
        i_connections = np.sum(np.abs(Aa[node_idx, i_indices]))

        # If ALL connections at this node are current sources, check KCL
        if i_connections == connections_per_node[node_idx]:
            kcl_sum = 0
            offending_sources = []  # Keep a list of the bad sources

            for idx in i_indices:
                if Aa[node_idx, idx] != 0:
                    val = cir_val[cir_el == br_el[idx]][0, 0]
                    kcl_sum += Aa[node_idx, idx] * val
                    offending_sources.append(br_el[idx])

            if kcl_sum != 0:
                # node_idx is the matrix row. nodes[node_idx] gives the physical node number!
                bad_node = nodes[node_idx]
                sources_str = ", ".join(offending_sources)

                sys.exit("ERROR: Current sources in series violate KCL at "
                         f"Node {bad_node}. Offending sources: {sources_str}")


def print_cir_info(cir_el, cir_nd, b, n, nodes, el_num):
    """
    Prints information about the circuit.

    Parameters
    ----------
    cir_el : numpy.ndarray
        Array of strings with the element names. Size: (e,)
    cir_nd : numpy.ndarray
        Array of integers with the nodes of the circuit. Size: (e, 4)
    b : int
        Total number of branches in the circuit.
    n : int
        Total number of unique nodes in the circuit.
    nodes : numpy.ndarray
        1D array of unique nodes in the circuit.
    el_num : int
        Total number of original elements in the circuit.

    Returns
    -------
    None.
        Prints directly to the console.
    """
    # Element info
    print(str(el_num) + ' Elements')
    # Node info
    print(str(n) + ' Different nodes: ' +
          str(nodes))
    # Branch info
    print("\n" + str(b) + " Branches: ")

    for i in range(1, b+1):
        indent = 12  # Number of blanks for indent
        string = ("\t" + str(i) + ". branch:\t" +
                  str(cir_el[i-1]) + "i".rjust(indent - len(cir_el[i-1])) +
                  str(i) + "v".rjust(indent - len(str(i))) + str(i) +
                  " = e" + str(cir_nd[i-1, 0]) +
                  " - e" + str(cir_nd[i-1, 1]))
        print(string)

    # Variable info
    print("\n" + str(2*b + (n-1)) + " variables: ")
    # print all the nodes but the first(0 because is sorted)
    for i in nodes[1:]:
        print("e"+str(i)+", ", end="", flush=True)
    for i in range(b):
        print("i"+str(i+1)+", ", end="", flush=True)
    # print all the branches but the last to close it properly
    # It works because the minuimum amount of branches in a circuit must be 2.
    for i in range(b-1):
        print("v"+str(i+1)+", ", end="", flush=True)
    print("v"+str(b))

    # IT IS RECOMMENDED TO USE THIS FUNCTION WITH NO MODIFICATION.
