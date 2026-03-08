# -*- coding: utf-8 -*-
"""
.. module:: zlel_p2.py
:synopsis: Module for 2nd lab of LNLC subject.
It contains the circuit parser, a branch builder, info printing
and some integrity checks.
.. moduleauthor:: Aitor Serrano Murua (aserrano052@ikasle.ehu.eus), Erik 
Elorduy Bravo (eelorduy005@ikasle.ehu.eus)
"""
import sys
import os
import numpy as np

if __name__ == "zlel.zlel_p2":
    import zlel.zlel_p1 as zl1
else:
    import zlel_p1 as zl1


def build_tableau(A, M, N, Us, b, n):
    """
    Assembles the Tableau matrix T and the right-hand side vector U.
    T * x = U

    Parameters
    ----------
    A : numpy.ndarray
        Reduced incidence matrix of the circuit. Size: (n-1, b).
    M : numpy.ndarray
        BCE Voltage matrix. Size: (b, b).
    N : numpy.ndarray
        BCE Current matrix. Size: (b, b).
    Us : numpy.ndarray
        BCE Source vector. Size: (b, 1).
    b : int
        Number of branches.
    n : int
        Number of nodes (including reference).

    Returns
    -------
    T : numpy.ndarray
        Assembled Tableau matrix. Size: (2b + n - 1, 2b + n - 1).
    U : numpy.ndarray
        Assembled right-hand side vector. Size: (2b + n - 1, 1).

    """
    # 1. Create the Identity matrix for the KVL row
    I = np.eye(b, dtype=float)

    # 2. Create the exact sizes of the zero matrices needed to fill the gaps
    # n-1 is the number of non-reference nodes
    Z_top_left = np.zeros((n-1, n-1), dtype=float)
    Z_top_mid = np.zeros((n-1, b), dtype=float)
    Z_mid_right = np.zeros((b, b), dtype=float)
    Z_bot_left = np.zeros((b, n-1), dtype=float)

    # 3. Stitch the Tableau matrix (T) together using np.block
    # [   0         0      A ]  -> KCL
    # [ -A^T        I      0 ]  -> KVL
    # [   0         M      N ]  -> BCE
    T = np.block([
        [Z_top_left,  Z_top_mid,  A],
        [-A.T,        I,          Z_mid_right],
        [Z_bot_left,  M,          N]
    ])

    # 4. Create the right-hand side zero vectors
    Z_u_top = np.zeros((n-1, 1), dtype=float)
    Z_u_mid = np.zeros((b, 1), dtype=float)

    # 5. Stack the U vector vertically
    U = np.vstack((Z_u_top, Z_u_mid, Us))

    return T, U


def simulate_tr(cir_sim, filename, br_el, br_val, br_ctr, b, n, A):
    """
    Performs Transient (.TR) analysis and saves to CSV. 

    Parameters
    ----------
    cir_sim : dict
        Dictionary containing simulation parameters.
    filename : string
        String with the original circuit filename (including path).
    br_el : numpy.ndarray
        Array of strings with the branch names.
    br_val : numpy.ndarray
        Array of floats with the branch element values.
    br_ctr : numpy.ndarray
        Array of strings with the control element names.
    b : int
        Total number of branches.
    n : int
        Total number of unique nodes.
    A : numpy.ndarray
        Reduced incidence matrix. Size: (n-1, b).

    Returns
    -------
    None.

    """
    start = cir_sim['tr']['start']
    end = cir_sim['tr']['end']
    step = cir_sim['tr']['step']

    # Generate the file path: <circuit_name>.tr inside "sims" folder
    filepath = save_sim_output(filename, "sims", ".tr")
    header = build_csv_header("t", b, n)

    with open(filepath, 'w') as file:
        print(header, file=file)

        # We use np.arange with a tiny buffer on 'end' to ensure the final step is included
        t = start
        while t <= end + (step / 10.0):
            # 1. Get physics matrices for this EXACT moment in time
            M, N, Us = zl1.build_bce(br_el, br_val, br_ctr, b, t=t)

            # 2. Build and solve Tableau
            T, U = build_tableau(A, M, N, Us, b, n)
            try:
                sol = np.linalg.solve(T, U)
            except np.linalg.LinAlgError:
                sys.exit("Error solving Tableau equation!")

            # 3. Flatten solution and insert the current time 't' at the beginning
            # sol.flatten() turns a column vector like [[1], [2]] into [1, 2]
            sol_flat = sol.flatten()
            csv_row = np.concatenate(([t, 0.0], sol_flat))

            # 4. Format to 9 decimal places and write to file
            sol_csv = ','.join(['%.9f' % num for num in csv_row])
            print(sol_csv, file=file)

            t = round(t + step, 10)  # Round to prevent floating point drift


def simulate_dc(cir_sim, filename, br_el, br_val, br_ctr, b, n, A):
    """
    Performs DC Sweep (.DC) analysis and saves to CSV.

    Parameters
    ----------
    cir_sim : dict
        Dictionary containing simulation parameters.
    filename : string
        String with the original circuit filename (including path).
    br_el : numpy.ndarray
        Array of strings with the branch names.
    br_val : numpy.ndarray
        Array of floats with the branch element values.
    br_ctr : numpy.ndarray
        Array of strings with the control element names.
    b : int
        Total number of branches.
    n : int
        Total number of unique nodes.
    A : numpy.ndarray
        Reduced incidence matrix. Size: (n-1, b).

    Returns
    -------
    None.

    """
    start = cir_sim['dc']['start']
    end = cir_sim['dc']['end']
    step = cir_sim['dc']['step']
    src_name = cir_sim['dc']['src']

    # Find the source in the branches array
    if src_name not in br_el:
        sys.exit(f"ERROR: .DC source '{
                 src_name}' not found in circuit branches.")

    src_idx = np.where(br_el == src_name)[0][0]

    # Determine header letter (V or I) based on source type
    tvi = 'V' if src_name[0].upper() in ['V', 'B', 'E', 'H'] else 'I'

    # Generate the file path: <circuit_name>_<source>.dc inside "sims"
    filepath = save_sim_output(filename, "sims", f"_{src_name}.dc")
    header = build_csv_header(tvi, b, n)

    # Store the original value so we don't permanently corrupt the array
    original_val = br_val[src_idx][0]

    with open(filepath, 'w') as file:
        print(header, file=file)

        sweep_val = start
        while sweep_val <= end + (step / 10.0):
            # 1. Overwrite the source's amplitude with the current sweep value
            br_val[src_idx][0] = sweep_val

            # 2. Build and solve (t=0 since it's DC)
            M, N, Us = zl1.build_bce(
                br_el, br_val, br_ctr, b, t=0.0, is_op=True)
            T, U = build_tableau(A, M, N, Us, b, n)
            try:
                sol = np.linalg.solve(T, U)
            except np.linalg.LinAlgError:
                sys.exit("Error solving Tableau equation!")

            # 3. Format and write
            sol_flat = sol.flatten()
            csv_row = np.concatenate(([sweep_val, 0.0], sol_flat))
            sol_csv = ','.join(['%.9f' % num for num in csv_row])
            print(sol_csv, file=file)

            sweep_val = round(sweep_val + step, 10)

    # Restore the original value just in case another analysis runs after this
    br_val[src_idx][0] = original_val


def print_solution(sol, b, n):
    """
    Prints the solution vector clearly. Inserts e0 = 0V at the beginning of the
    node voltages.

    Parameters
    ----------
    sol : numpy.ndarray
        np array with the solution of the Tableau equations.
    b : int
        Number of branches.
    n : int
        Number of nodes.

    Returns
    -------
    None.

    """
    # The instructor solution needs to be a numpy array of numpy arrays of
    # float. If it is not, convert it to this format.
    if sol.dtype == np.float64:
        np.set_printoptions(sign=' ')  # Only from numpy 1.14
        tmp = np.zeros([np.size(sol), 1], dtype=float)
        for ind in range(np.size(sol)):
            tmp[ind] = np.array(sol[ind])
        sol = tmp

    tolerance = -1e-9  # Adjust as needed
    print("\n========== Nodes voltage to reference ========")
    for i in range(1, n):
        value = sol[i-1][0]
        print("e" + str(i) + " = ", "[{:10.9f}]".format(0.0 if value <= 0 and
                                                        value >= tolerance else value))
    print("\n========== Branches voltage difference ========")
    for i in range(1, b+1):
        value = sol[i+n-2][0]
        print("v" + str(i) + " = ", "[{:10.9f}]".format(0.0 if value <= 0 and
                                                        value >= tolerance else value))
    print("\n=============== Branches currents ==============")
    for i in range(1, b+1):
        value = sol[i+b+n-2][0]
        print("i" + str(i) + " = ", "[{:10.9f}]".format(0.0 if value <= 0 and
                                                        value >= tolerance else value))
    print("\n================= End solution =================\n")


def build_csv_header(tvi, b, n):
    """
    This function build the csv header for the output files. First column will 
    be v or i if .dc analysis or t if .tr and it will be given by argument tvi.
    The header will have this form: t/v/i,e_1,..,e_n-1,v_1,..,v_b,i_1,..i_b.

    Parameters
    ----------
    tvi : string
        "v" or "i" if .dc analysis or "t" if .tran.
    b : int
        Number of branches.
    n : int
        Number of nodes.

    Returns
    -------
    header : string
        The header in csv format as string.

    """
    header = tvi + ",e0"
    for i in range(1, n):
        header += ",e" + str(i)
    for i in range(1, b+1):
        header += ",v" + str(i)
    for i in range(1, b+1):
        header += ",i" + str(i)
    return header


def save_sim_output(filename, sims_folder_name, extension):
    """
    This function creates an absolute path to a filename inserting a folder 
    given by "sims_folder_name" and changing its extension by another given by 
    "extensión" (. needs to be included).

    Parameters
    ----------
    filename : string
        String with the filename (incluiding the path).
    sims_folder_name : string
        String with the name of the folder to save the sims.
    extension : string
        New extension for the file.

    Returns
    -------
    new_file_path : string
        The absolute file path with the inserted sims_folder_name and new 
        extension.

    """

    if not os.path.exists(filename):
        print("file does not exist")
        return
    filename = os.path.abspath(filename)
    dir_path = os.path.dirname(filename)
    base_name, ext = os.path.splitext(os.path.basename(filename))
    new_dir_path = os.path.join(dir_path, sims_folder_name)
    try:
        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path)
    except OSError as e:
        print(f"Error creating directory: {e}")
        return

    new_filename = f"{base_name}{extension}"
    new_file_path = os.path.join(new_dir_path, new_filename)
    return new_file_path
