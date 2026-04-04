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
    This function takes a .cir test circuit and parses it to 4 matrices and a
    dictionary. It separates circuit elements from simulation commands.

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
    cir_sim : dict
        Dictionary containing the simulation commands and parameters. 
        Keys are strings ('pr', 'op', 'dc', 'tr'). Values are booleans for 
        'pr' and 'op', and nested dictionaries for 'dc' and 'tr' containing 
        the sweep parameters (e.g., {'start': 0.0, 'end': 10.0, 'step': 0.1}).
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

        # Split elements and simulations using NumPy masks
        sim_mask = np.char.startswith(data[:, 0], '.')

        # Check if there is analysis info. Exit if not.
        if not np.any(sim_mask):
            sys.exit("ERROR: No analysis commands (.OP, .DC, etc.) found in "
                     f"{filename}.")

        elem_data = data[~sim_mask]  # Only the physical components
        sim_data = data[sim_mask]    # Only the simulation commands

        cir_nd = elem_data[:, 1:5].astype(int)
        if not np.any(cir_nd == 0):
            print("ERROR: The circuit does not have a reference node (0).")
            sys.exit(1)

        cir_el = elem_data[:, 0]
        if len(cir_el) != len(np.unique(cir_el)):
            sys.exit("ERROR: There are duplicate element names in the circuit "
                     "definition.")

        cir_val = elem_data[:, 5:8].astype(float)
        cir_ctr = elem_data[:, 8]

        cir_sim = {}  # Save the simulation commands (sim_data)

        for row in sim_data:
            # Convert to lowercase to catch '.OP' or '.op'
            cmd = row[0].lower()

            if cmd == '.pr':
                cir_sim['pr'] = True
            elif cmd == '.op':
                cir_sim['op'] = True
            elif cmd == '.dc':
                # Syntax: .DC 0 0 0 0 start end step source
                cir_sim['dc'] = {
                    'start': float(row[5]),
                    'end': float(row[6]),
                    'step': float(row[7]),
                    'src': row[8]
                }
            elif cmd == '.tr':
                # Syntax: .TR 0 0 0 0 start end step 0
                cir_sim['tr'] = {
                    'start': float(row[5]),
                    'end': float(row[6]),
                    'step': float(row[7])
                }

        return cir_el, cir_nd, cir_val, cir_ctr, cir_sim

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


def build_branches(cir_el, cir_nd, cir_val, cir_ctr):
    """
    Expands the elements into branches for the digraph.

    Parameters
    ----------
    cir_el : numpy.ndarray
        Array of strings with the element names. Size: (e,)
    cir_nd : int
        Array of integers with the nodes of the circuit. Size: (e, 4)

    Returns
    -------
    br_el : numpy.ndarray
        Array of strings with the expanded branch names (e.g., 'Q_1_be').
    br_nd : numpy.ndarray
        Array of integers with the branch nodes (positive and negative 
        terminals). Size: (b, 2).
    br_val : numpy.ndarray
        Array of floats containing the values associated with each branch. 
        Copied directly from the parent element. Size: (b, 3).
    br_ctr : numpy.ndarray
        Array of strings with the control element names associated with 
        each branch. Copied directly from the parent element. Size: (b,).
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

    # 3. Initialize fixed-size NumPy arrays
    # '<U50': a string of up to 50 characters (needed to append '_be', etc.)
    br_el = np.empty(b, dtype='<U50')
    br_nd = np.empty((b, 2), dtype=int)
    br_val = np.empty((b, 3), dtype=float)
    br_ctr = np.empty(b, dtype='<U50')

    # 4. Fill the arrays based on the topology rules
    idx = 0
    for i in range(len(cir_el)):
        el = cir_el[i]
        nd = cir_nd[i]
        val = cir_val[i]
        ctr = cir_ctr[i]

        if q_mask[i]:
            # BJT rules: Nodes are defined as [Collector, Base, Emitter] ->
            # [nd[0], nd[1], nd[2]]
            br_el[idx] = el + "_be"
            br_nd[idx] = [nd[1], nd[2]]  # Base to Emitter
            br_val[idx] = val
            br_ctr[idx] = ctr
            idx += 1

            br_el[idx] = el + "_bc"
            br_nd[idx] = [nd[1], nd[0]]  # Base to Collector
            br_val[idx] = val
            br_ctr[idx] = ctr
            idx += 1

        elif a_mask[i]:
            # OpAmp rules: Nodes are defined as [N+, N-, N_out, N_ref] ->
            # [nd[0], nd[1], nd[2], nd[3]]
            br_el[idx] = el + "_in"
            br_nd[idx] = [nd[0], nd[1]]  # N+ to N-
            br_val[idx] = val
            br_ctr[idx] = ctr
            idx += 1

            br_el[idx] = el + "_ou"
            br_nd[idx] = [nd[2], nd[3]]  # N_out to N_ref
            br_val[idx] = val
            br_ctr[idx] = ctr
            idx += 1

        else:
            # Standard 2-terminal components
            br_el[idx] = el
            br_nd[idx] = [nd[0], nd[1]]  # Positive to Negative
            br_val[idx] = val
            br_ctr[idx] = ctr
            idx += 1

    return br_el, br_nd, br_val, br_ctr, b


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
    # Since our 'nodes' array is sorted, the reference node '0' is ALWAYS at
    # row 0.
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
                 " two nodes.")

    # 2. Floating nodes
    connections_per_node = np.sum(np.abs(Aa), axis=1)
    if np.any(connections_per_node == 1):
        sys.exit("ERROR: The ciruit has a node(s) with a single connection "
                 "(floating nodes).")

    # 3. Parallel voltage sources
    # Find all branches starting with 'V' or 'E'
    br_lower = np.char.lower(br_el)
    v_mask = (np.char.startswith(br_lower, 'v') |
              np.char.startswith(br_lower, 'e') |
              np.char.startswith(br_lower, 'h') |
              np.char.startswith(br_lower, 'b'))
    # Gets the column numbers for the voltage sources
    v_indices = np.flatnonzero(v_mask)

    if len(v_indices) > 1:
        # Extract only the columns of Aa that are voltage sources
        Aa_v = Aa[:, v_indices]

        # Calculate the correlation matrix: V^T * V
        correlation_matrix = np.dot(Aa_v.T, Aa_v)

        # Set the diagonal to 0 (so a source isn't flagged as parallel with
        # itself!)
        np.fill_diagonal(correlation_matrix, 0)

        # Find where the dot product is 2 (parallel) or -2 (anti-parallel)
        # np.argwhere gives us a list of [row, col] pairs where the value is 2
        # or -2
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

            # If they are connected backwards (-2), flip val2 to compare them
            # fairly
            if correlation_matrix[idx1, idx2] == -2:
                val2 = -val2

            if val1 != val2:
                sys.exit(f"Parallel V sources at branches {br1} and {br2}.")

    # 3. SERIES CURRENT SOURCES
    i_mask = (np.char.startswith(br_lower, 'i') |
              np.char.startswith(br_lower, 'g') |
              np.char.startswith(br_lower, 'f') |
              np.char.startswith(br_lower, 'y'))
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
                # node_idx is the matrix row. nodes[node_idx] gives the
                # physical node number!
                bad_node = nodes[node_idx]
                sources_str = ", ".join(offending_sources)

                sys.exit("ERROR: Current sources in series violate KCL at "
                         f"Node {bad_node}. Offending sources: {sources_str}")


def build_bce(br_el, br_val, br_ctr, b, t=0.0, is_op=False, v_j=None, x_k=None,
              h=0.0, n=0):
    """
    Builds the Branch Constitutive Equation (BCE) matrices: M, N, and Us. It
    can handle non-linear elements using Newton-Raphson.
    Equation format: M*v + N*i = Us

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
        Total number of branches in the circuit.
    t : float, optional
        The current time step for transient analysis. Used to calculate the 
        instantaneous value of sinusoidal sources. The default is 0.0.
    is_op : boolean, optional
        Flag to check if we are doing an .op analysis.
    v_j : numpy.ndarray, optional
        Array of branch voltages at the current NR iteration. Defaults to
        None.
    x_k : nupmpy.ndarray, optional
        Column vector containing the Tableau solution from the previous time 
        step (k). Used to extract historical voltages/currents for dynamic 
        elements (C, L). Defaults to None.
    h : float, optional
        Time step size of the transient simulation. Used to calculate the 
        discrete equivalents of dynamic elements. Defaults to 0.0
    n : integer, optional
        Total number of unique nodes in the circuit. Used to calculate the 
        index offsets when extracting values from the x_k vector. Defaults to 0.

    Returns
    -------
    M : numpy.ndarray
        BCE Voltage matrix representing the voltage coefficients. Size: (b, b).
    N : numpy.ndarray
        BCE Current matrix representing the current coefficients. Size: (b, b).
    Us : numpy.ndarray
        BCE Source vector representing the independent source values. Size: 
        (b, 1).

    """

    # Initialize empty matrices with floats
    M = np.zeros((b, b), dtype=float)
    N = np.zeros((b, b), dtype=float)
    Us = np.zeros((b, 1), dtype=float)

    for i in range(b):
        branch_name = br_el[i]
        val = br_val[i]
        ctr = br_ctr[i]

        type_letter = branch_name[0].upper()

        # --- RESISTORS ---
        if type_letter == 'R':
            M[i, i] = 1.0
            N[i, i] = -val[0]

        # --- INDEPENDENT VOLTAGE SOURCES ---
        elif type_letter in ['V', 'B']:
            M[i, i] = 1.0
            if type_letter == 'V' or is_op:
                Us[i, 0] = val[0]
            else:
                # B_xx (Sinusoidal Voltage)
                Us[i, 0] = val[0] * np.sin(2 * np.pi * val[1] * t +
                                           (np.pi / 180.0) * val[2])

        # --- INDEPENDENT CURRENT SOURCES ---
        elif type_letter in ['I', 'Y']:
            N[i, i] = 1.0
            if type_letter == 'I' or is_op:
                Us[i, 0] = val[0]
            else:
                # Y_xx (Sinusoidal Current)
                Us[i, 0] = val[0] * \
                    np.sin(2 * np.pi * val[1] * t + (np.pi / 180.0) * val[2])

        # --- CONTROLLED SOURCES (E, H, G, F) ---
        elif type_letter in ['E', 'H', 'G', 'F']:
            # Find the row index of the controlling branch instantly
            ctrl_idx = np.where(np.char.lower(br_el) == str(ctr).lower())[0][0]

            if type_letter == 'E':    # VCVS
                M[i, i] = 1.0
                M[i, ctrl_idx] = -val[0]
            elif type_letter == 'H':  # CCVS
                M[i, i] = 1.0
                N[i, ctrl_idx] = -val[0]
            elif type_letter == 'G':  # VCCS
                N[i, i] = 1.0
                M[i, ctrl_idx] = -val[0]
            elif type_letter == 'F':  # CCCS
                N[i, i] = 1.0
                N[i, ctrl_idx] = -val[0]

        # --- IDEAL OP-AMPS ---
        elif type_letter == 'A':
            if branch_name.endswith("_in"):
                M[i, i] = 1.0  # v_in = 0
            elif branch_name.endswith("_ou"):
                # Force the input current to be 0 using the output branch's
                # equation row
                # The input branch is always exactly 1 index before the output.
                in_idx = i - 1
                N[i, in_idx] = 1.0  # i_in = 0

        # --- NON-LINEAR: DIODE ---
        elif type_letter == 'D':
            import zlel.zlel_p3 as zl3  # Local import to avoid circular
            # dependency
            I0, nD = val[0], val[1]
            Vdj = v_j[i]

            gd, Id = zl3.diode_NR(I0, nD, Vdj)

            M[i, i] = gd
            N[i, i] = 1.0
            Us[i, 0] = Id

        # --- NON-LINEAR: BJT NPN ---
        elif type_letter == 'Q':
            import zlel.zlel_p3 as zl3
            Ies, Ics, betaF = val[0], val[1], val[2]

            # Since BOTH iE and iC are defined as flowing OUT of the transistor
            # in bjt_NR and our branches _be and _bc also flow OUT,  N = +1.0
            # for both. Also, build_branches creates _be and _bc are created
            # concecutively.
            if branch_name.endswith("_be"):
                idx_be = i
                idx_bc = i + 1
                Vbej, Vbcj = v_j[idx_be], v_j[idx_bc]

                g11, g12, _, _, IE, _ = zl3.bjt_NR(Ies, Ics, betaF, Vbej, Vbcj)

                # Equation: iE + g11*VBE + g12*VBC = IE
                N[i, i] = 1.0
                M[i, i] = g11
                M[i, idx_bc] = g12
                Us[i, 0] = IE
            else:
                idx_bc = i
                idx_be = i - 1
                Vbej, Vbcj = v_j[idx_be], v_j[idx_bc]

                _, _, g21, g22, _, IC = zl3.bjt_NR(Ies, Ics, betaF, Vbej, Vbcj)

                # Equation: iC + g21*VBE + g22*VBC = IC
                N[i, i] = 1.0
                M[i, idx_be] = g21
                M[i, i] = g22
                Us[i, 0] = IC

        # --- DYNAMIC: CAPACITOR ---
        elif type_letter == 'C':
            if t == 0.0:
                # 1st Iteration: Acts as an ideal voltage source using Vc,0
                M[i, i] = 1.0
                Us[i, 0] = val[1]  # initial voltage
            else:
                # Backward Euler Equivalent:
                # v_{c,k+1} - (h/C)*i_{c,k+1} = v_{c,k}
                v_ck = x_k[n - 1 + i, 0]  # Extract previous voltage from x_k
                M[i, i] = 1.0
                N[i, i] = -(h / val[0])   # val[0] is Capacitance (C)
                Us[i, 0] = v_ck

        # --- DYNAMIC: INDUCTOR ---
        elif type_letter == 'L':
            if t == 0.0:
                # 1st Iteration: Acts as an ideal current source using Il,0
                N[i, i] = 1.0
                Us[i, 0] = val[1]  # val[1] is the initial current
            else:
                # Backward Euler Equivalent:
                # v_{L,k+1} - (L/h)*i_{L,k+1} = -(L/h)*i_{L,k}
                i_lk = x_k[n - 1 + b + i, 0]  # Extract previous current in x_k
                M[i, i] = 1.0
                N[i, i] = -(val[0] / h)       # val[0] is Inductance (L)
                Us[i, 0] = -(val[0] / h) * i_lk

    return M, N, Us


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
