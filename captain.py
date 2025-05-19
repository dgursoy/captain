# #########################################################################
# Copyright (c) 2025, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2025. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""
Module for generating and manipulating coded aperture sequences.
"""

__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2024, UChicago Argonne, LLC"
__license__ = "Argonne Open Source License (BSD-style)"
__version__ = "0.1.0"
__docformat__ = "restructuredtext en"
__all__ = ['generate_binary_sequence',
           'generate_physical_aperture',
           'generate_beam',
           'generate_kernel_matrix',
           'get_aperture_subset',
           'apply_poisson_noise',
           'reconstruct_beam']

import random
import numpy as np
from scipy import signal, optimize
from scipy.ndimage import shift
import xraydb

# Set fixed random seed for reproducibility
np.random.seed(0)
random.seed(0)

def generate_beam(maxsize, avgsize, alpha=0.0, intensity=1.0):
    """
    Generate a beam profile for coded aperture (tukey function).
    
    Args:
        maxsize (int): Total length of the sequence
        avgsize (int): Width of the beam
        alpha (float): Shape parameter of the beam
        intensity (float): Peak intensity of the beam
        
    Returns:
        ndarray: Beam profile sequence
    """
    _sig = np.zeros(maxsize, dtype='float32')
    first = int((maxsize - 1) * 0.5 - avgsize * 0.5)
    beam = signal.windows.tukey(int(avgsize), alpha=alpha)
    beam = beam * (intensity / np.max(beam))  # Scale to desired intensity
    _sig[first:first+int(avgsize)] = beam
    return _sig

def generate_sequence(length, bits_for_zero=10, bits_for_one=10):
    """
    Generate a random binary sequence and convert it to an aperture sequence.
    
    Args:
        length (int): Length of the initial binary sequence
        bits_for_zero (int): Number of bits to represent a '0'
        bits_for_one (int): Number of bits to represent a '1'
    
    Returns:
        ndarray: The expanded aperture sequence
    """
    # Generate random binary sequence
    sequence = [random.randint(0, 1) for _ in range(length)]
    
    # Convert to aperture sequence
    aperture = []
    for bit in sequence:
        if bit == 0:
            aperture.extend([0] * bits_for_zero)
        else:
            aperture.extend([1] * bits_for_one)
            
    return np.array(aperture)

def generate_physical_aperture(sequence, material='Au', thickness=100, energy=20, 
                             density=19.32, angle=0.0):
    """
    Convert a binary aperture sequence to transmission values based on material properties.
    
    Args:
        sequence (array-like): Binary sequence where 0 means material present
        material (str): Material name (e.g., 'Au', 'W', 'Pt')
        thickness (float): Material thickness in micrometers
        energy (float): X-ray energy in keV
        density (float): Material density in g/cm³
        
    Returns:
        ndarray: Aperture sequence with actual transmission values
        where 0 means material present (transmission=exp(-μt)) and 1 means open (transmission=1)
    """
    # Convert input to numpy array if it isn't already
    binary_aperture = np.asarray(sequence)
    
    # Calculate attenuation coefficient (mu)
    mu = xraydb.mu_elam(material, energy * 1e3)  # cm²/g
    
    # Calculate transmission through material
    attenuation = mu * density * thickness * 1e-4
    
    # Convert binary sequence to transmission values
    # Where aperture is 0: material present (transmission = exp(-mu*t))
    # Where aperture is 1: no material (transmission = 1)
    transmission = np.where(binary_aperture == 0, 
                          np.exp(-attenuation),
                          1.0)
    
    return transmission

def get_aperture_subset(aperture, position, subset_size):
    """
    Extract a subset of the aperture sequence starting at position with linear interpolation.
    
    Args:
        aperture (ndarray): Full aperture sequence
        position (float): Starting position for the subset
        subset_size (int): Length of the subset to extract
        
    Returns:
        ndarray: Interpolated subset of the aperture sequence
    """
    shifted = shift(aperture, -position, mode='constant', cval=0.0, order=1)
    return shifted[:subset_size]

def generate_data(aperture_subset, beam):
    """
    Calculate dot product of aperture subset with beam profile.
    
    Args:
        aperture_subset (ndarray): Subset of aperture sequence
        beam (ndarray): Beam profile
        
    Returns:
        float: Dot product of aperture subset and beam
    """
    return np.dot(aperture_subset, beam)

def generate_data_sequence(aperture, positions, beam):
    """
    Generate measurement sequence using coded aperture.
    
    Args:
        aperture (ndarray): Full aperture sequence
        positions (ndarray): List of positions to generate measurements for
        beam (ndarray): Beam profile
        
    Returns:
        ndarray: List of measurements, one for each position
    """
    return [generate_data(get_aperture_subset(aperture, pos, len(beam)), beam) 
            for pos in positions]

def generate_kernel_matrix(positions, aperture, beam_length, intensities=None):
    """
    Generate kernel matrix from aperture positions and intensities.
    
    Args:
        positions (ndarray): Array of measurement positions
        aperture (ndarray): The aperture sequence
        beam_length (int): Length of the beam profile
        intensities (ndarray, optional): Intensity coefficients for each position. 
                                       If None, uses ones. Default is None.
    
    Returns:
        ndarray: Kernel matrix where each row represents aperture at different position,
                scaled by corresponding intensity
    """
    if intensities is None:
        intensities = np.ones(len(positions))
    
    # Generate base kernel matrix
    kernel_matrix = np.zeros((len(positions), beam_length))
    for i, pos in enumerate(positions):
        kernel_matrix[i] = get_aperture_subset(aperture, pos, beam_length)
        # Multiply each row by its corresponding intensity
        kernel_matrix[i] *= intensities[i]
    
    return kernel_matrix

def reconstruct_beam(kernel_matrix, measurements, alpha=0.0):
    """
    Reconstruct beam profile from measurements using non-negative least squares.
    
    Args:
        kernel_matrix (ndarray): Matrix where each row is the aperture subset at different positions
        measurements (ndarray): Array of measurements
        alpha (float, optional): Regularization parameter. Default is 0.0
        
    Returns:
        ndarray: Reconstructed beam profile
    """
    K = kernel_matrix
    
    # Use pseudo-inverse for better numerical stability
    KTK = np.dot(K.T, K)
    reg_matrix = KTK + alpha * np.eye(K.shape[1])
    KTy = np.dot(K.T, measurements)
    
    # Then solve with NNLS
    try:
        reconstructed = optimize.nnls(reg_matrix, KTy, maxiter=2000)[0]
    except:
        print("Warning: NNLS failed, using regularized least squares solution")
        reconstructed = np.linalg.lstsq(reg_matrix, KTy, rcond=None)[0]
    
    return reconstructed

def apply_poisson_noise(measurements):
    """
    Apply Poisson noise to measurements.
    
    Args:
        measurements (ndarray): Clean measurements
        
    Returns:
        ndarray: Noisy measurements
    """
    return np.random.poisson(measurements)
