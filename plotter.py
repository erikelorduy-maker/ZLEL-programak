# -*- coding: utf-8 -*-
"""
.. module:: plotter.py
:synopsis: Module for plotting the circuit simulations.  
.. moduleauthor:: Aitor Serrano Murua (aserrano052@ikasle.ehu.eus), Erik 
Elorduy Bravo (eelorduy005@ikasle.ehu.eus)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys


def plot_from_csv(filename, x_col, y_cols):
    """
    Reads a simulation CSV and plots multiple variables on the y-axis.
    """
    try:
        # Read the CSV. names=True uses the first row as headers.
        data = np.genfromtxt(filename, delimiter=',', names=True)
    except OSError:
        sys.exit(f"File {filename} not found!")

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)

    # Allow plotting multiple y-columns at once!
    for y in y_cols:
        if y in data.dtype.names:
            ax1.plot(data[x_col], data[y], label=f'{y}')
        else:
            print(f"Warning: Column '{y}' not found in {filename}.")

    ax1.set_xlabel(x_col)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Simulation Results: {filename}")
    ax1.legend()
    ax1.grid(True)
    plt.show()


if __name__ == "__main__":
    # Example usage: Plot time vs. node 1 and node 2 voltages
    plot_from_csv("cirs/sims/3_zlel_RLC.tr", "t", ["e2", "e3"])
