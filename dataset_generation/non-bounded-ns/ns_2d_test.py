import torch

import math

import matplotlib.pyplot as plt
import matplotlib

from random_fields import GaussianRF
from einops import rearrange, repeat

from timeit import default_timer

import scipy.io
import numpy as np

import scipy.io
import tqdm
from scipy.io import savemat
from tqdm import tqdm


# w0: initial vorticity
# f: forcing term
# visc: viscosity (1/Re)
# T: final time
# delta_t: internal time-step for solve (descrease if blow-up)
# record_steps: number of in-time snapshots to record
def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):

    # Grid size - must be power of 2
    N = w0.size()[-1]

    # Maximum frequency
    k_max = math.floor(N / 2.0)

    # Number of steps to final time
    steps = math.ceil(T / delta_t)

    # Initial vorticity to Fourier space - using new FFT API
    w_h = torch.fft.fftn(w0, dim=[1, 2], norm='backward')

    # Forcing to Fourier space
    f_h = torch.fft.fftn(f, dim=(-2, -1), norm='backward')

    # If same forcing for the whole batch
    if len(f_h.shape) < len(w_h.shape):
        f_h = rearrange(f_h, '... -> 1 ...')

    # Record solution every this number of steps
    record_time = math.floor(steps / (record_steps))

    # Wavenumbers in y-direction
    k_y = torch.cat((
        torch.arange(start=0, end=k_max, step=1, device=w0.device),
        torch.arange(start=-k_max, end=0, step=1, device=w0.device)),
        0).repeat(N, 1)    # Wavenumbers in x-direction
    k_x = k_y.transpose(0, 1)
    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0

    if isinstance(visc, np.ndarray):
        visc = torch.from_numpy(visc).to(w0.device)
        visc = repeat(visc, 'b -> b m n', m=N, n=N)
        lap = repeat(lap, 'm n -> b m n', b=w0.shape[0])

    # Dealiasing mask
    dealias = torch.unsqueeze(
        torch.logical_and(
            torch.abs(k_y) <= (2.0 / 3.0) * k_max,
            torch.abs(k_x) <= (2.0 / 3.0) * k_max
        ).float(), 0)
    
    # Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    # Record counter
    c = 0
    # Physical time
    t = 0.0
    
    for j in tqdm(range(steps)):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        # Velocity field in x-direction = psi_y
        q = psi_h.clone()
        q_real_temp = q.real.clone()
        q.real = -2 * math.pi * k_y * q.imag
        q.imag = 2 * math.pi * k_y * q_real_temp
        q = torch.fft.ifftn(q, dim=[1, 2], norm='backward').real

        # Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        v_real_temp = v.real.clone()
        v.real = 2 * math.pi * k_x * v.imag
        v.imag = -2 * math.pi * k_x * v_real_temp
        v = torch.fft.ifftn(v, dim=[1, 2], norm='backward').real

        # Partial x of vorticity
        w_x = w_h.clone()
        w_x_temp = w_x.real.clone()
        w_x.real = -2 * math.pi * k_x * w_x.imag
        w_x.imag = 2 * math.pi * k_x * w_x_temp
        w_x = torch.fft.ifftn(w_x, dim=[1, 2], norm='backward').real

        # Partial y of vorticity
        w_y = w_h.clone()
        w_y_temp = w_y.real.clone()
        w_y.real = -2 * math.pi * k_y * w_y.imag
        w_y.imag = 2 * math.pi * k_y * w_y_temp
        w_y = torch.fft.ifftn(w_y, dim=[1, 2], norm='backward').real

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.fftn(q * w_x + v * w_y,
                             dim=[1, 2], norm='backward')

        # Dealias
        F_h *= dealias

        # Cranck-Nicholson update
        factor = 0.5 * delta_t * visc * lap
        num = -delta_t * F_h + delta_t * f_h + (1.0 - factor) * w_h
        w_h = num / (1.0 + factor)

        # Update real time (used only for recording)
        t += delta_t

        if (j + 1) % record_time == 0:
            # Solution in physical space
            w = torch.fft.ifftn(w_h, dim=[1, 2], norm='backward').real
            if w.isnan().any().item():
                raise ValueError('NaN values found.')

            # Record solution and time
            sol[..., c] = w
            sol_t[c] = t

            c += 1

    return sol, sol_t




device = torch.device("cuda")

s = 128
sub = 1
N = 1000  # Generate 1000 samples for testing
N_vis = 10  # First 10 samples will be saved with all timesteps for visualization
GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

t = torch.linspace(0, 1, s + 1, device=device)
t = t[0:-1]
X, Y = torch.meshgrid(t, t)
f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))
record_steps = 50  # Generate 50 timesteps
bsize = 200

print("=" * 70)
print("GENERATING TEST DATA + VISUALIZATION DATA IN SINGLE RUN")
print(f"- Test data: {N} samples with 2 timesteps (25th, 50th)")
print(f"- Visualization data: First {N_vis} samples with all 50 timesteps")
print("=" * 70)

# Allocate arrays
u_selected = torch.zeros(N, s, s, 2)  # For ns-nonbounded.mat (all 1000 samples, 2 timesteps)
u_vis = torch.zeros(N_vis, s, s, record_steps)  # For ns-nonbounded-vis.mat (first 10 samples, 50 timesteps)

c = 0
t0 = default_timer()

for j in range(N // bsize):
    print(f"\nBatch {j+1}/{N//bsize}")
    w0 = GRF.sample(bsize)
    sol, sol_t = navier_stokes_2d(w0, f, 1e-3, 5, 1e-3, record_steps)  # T=5 for physical time = 5
    
    # For ALL samples: extract only 25th and 50th timesteps for ns-nonbounded.mat
    u_selected[c : (c + bsize), :, :, 0] = sol[:, :, :, 24]  # 25th timestep
    u_selected[c : (c + bsize), :, :, 1] = sol[:, :, :, 49]  # 50th timestep
    
    # For FIRST 10 samples: save all 50 timesteps for ns-nonbounded-vis.mat
    if c < N_vis:
        end_idx = min(c + bsize, N_vis)
        num_vis_samples = end_idx - c
        u_vis[c:end_idx, :, :, :] = sol[:num_vis_samples, :, :, :]
        print(f"  → Saved visualization data for samples {c} to {end_idx-1}")
    
    c += bsize
    t1 = default_timer()
    print(f"  Progress: {c}/{N}, Time: {t1 - t0:.2f}s")

# Save test data (1000 samples, 2 timesteps)
filename_test = "ns-nonbounded.mat"
scipy.io.savemat(filename_test, mdict={
    "u": u_selected.cpu().numpy(),  # Shape: [1000, 128, 128, 2]
    "t": sol_t[[24, 49]].cpu().numpy()  # Times for 25th and 50th steps
})
print(f"\n✓ Saved: {filename_test} ({u_selected.numel() * 8 / 1024**2:.1f} MB)")

# Save visualization data (first 10 samples, all 50 timesteps)
filename_vis = "ns-nonbounded-vis.mat"
scipy.io.savemat(filename_vis, mdict={
    "u": u_vis.cpu().numpy(),  # Shape: [10, 128, 128, 50]
    "t": sol_t.cpu().numpy()  # All 50 timesteps
})
print(f"✓ Saved: {filename_vis} ({u_vis.numel() * 8 / 1024**2:.1f} MB)")

del u_selected
del u_vis

print("\n" + "="*70)
print("DATA GENERATION COMPLETE!")
print(f"- {filename_test}: 1000 samples with 2 timesteps → Move to data/DiffPDE/testing/")
print(f"- {filename_vis}: 10 samples with 50 timesteps → Use with visualize.py")
print("- First 10 samples are IDENTICAL in both files")
print("="*70)

