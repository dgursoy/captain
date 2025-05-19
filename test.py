import captain
import matplotlib.pyplot as plt
import numpy as np

########################################################
# Batch parameters
########################################################
trials = 100 # Number of trials for confidence interval
positions = np.arange(0, 200, 1) # Number of aperturepositions
intensity = 1000 # Intensity of the beam
exposures = np.random.uniform(low=1, high=1, size=len(positions)) # Exposure scale for each position

########################################################
# Preparing the beam and aperture
########################################################
beam = captain.generate_beam(
    maxsize=100, # length of the beam
    avgsize=20, # average size of the beam
    alpha=0.5, # shape parameter of the beam
) 
sequence = captain.generate_sequence(
    length=256, # length of the sequence
    bits_for_zero=10, # number of bits for 0 [mu]
    bits_for_one=10 # number of bits for 1 [mu]
)
aperture = captain.generate_physical_aperture(
    sequence=sequence,
    material='Au', # material of the aperture
    thickness=10, # thickness of the aperture
    energy=10, # energy of the beam
)

########################################################
# Generate kernel matrix and measurements
########################################################
kernel_matrix = captain.generate_kernel_matrix(positions, aperture, len(beam), exposures, intensity)
noisefree_measurements = np.dot(kernel_matrix, beam)

########################################################
# Trials for confidence interval
########################################################
all_reconstructions = np.zeros((trials, len(beam)))
for i in range(trials):
    noisy_measurements = captain.apply_poisson_noise(noisefree_measurements)
    reconstructed_beam = captain.reconstruct_beam(kernel_matrix, noisy_measurements)
    all_reconstructions[i] = reconstructed_beam
mean_reconstruction = np.mean(all_reconstructions, axis=0)
std_reconstruction = np.std(all_reconstructions, axis=0)

########################################################
# Plot results
########################################################
plt.figure(figsize=(6, 6))

# Kernel matrix (top)
plt.subplot(311)
plt.imshow(kernel_matrix, aspect='auto', cmap='viridis')
plt.title('Kernel Matrix')
plt.xlabel('Beam position')
plt.ylabel('Measurement number')
plt.colorbar(label='Aperture value')
# Add legend for aperture values
plt.plot([], [], color='yellow', label='1: Transparent')  # Light color from viridis
plt.plot([], [], color='darkblue', label='0: Opaque')    # Dark color from viridis
plt.legend(loc='upper right')

# Measurements (middle)
plt.subplot(312)
plt.step(np.arange(len(noisy_measurements)), noisy_measurements, 'tab:blue', where='mid', label='Noisy measurements')
plt.step(np.arange(len(noisefree_measurements)), noisefree_measurements, 'tab:red', where='mid', label='Noisefree measurements', alpha=0.8)
plt.title('Measurements')
plt.legend()
plt.grid(True)

# Beam reconstruction (bottom)
plt.subplot(313)
x = np.arange(len(mean_reconstruction))
plt.step(x, mean_reconstruction, 'tab:blue', where='mid', label='Reconstructed beam')
plt.step(x, beam, 'tab:red', where='mid', label='Original beam', alpha=0.8)

# Add confidence interval if std is available
if std_reconstruction is not None:
    # Create stepped version of upper and lower bounds
    upper = mean_reconstruction + 2*std_reconstruction
    lower = np.maximum(0, mean_reconstruction - 2*std_reconstruction)  # Clip at zero
    
    # Plot stepped fill_between
    plt.fill_between(x, lower, upper,
                    step='mid',  # Match the step style
                    color='tab:blue', alpha=0.2,
                    label='95% confidence')

plt.title('Beam Reconstruction')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()