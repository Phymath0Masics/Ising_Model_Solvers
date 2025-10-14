"""
Ising Model Solvers: DOCH, ADOCH, SA, BSB, SimCIM, SIS

This module implements multiple solvers for the Ising model:
    - DOCH (Difference Of Convex Hamiltonian)
    - ADOCH (Accelerated DOCH)
    - SA (Simulated Annealing)
    - BSB (Ballistic Spin/Bifurcation machine)
    - SimCIM (Simulated Coherent Ising Machine)
    - SIS (Spring-damping-based Ising Machine)
    - Utility functions for generating small and large random Ising matrices and computing norms.

All solvers expose a solve(...) method returning (energies, times, final_spins).
"""

# import necessary libraries
import os
import time
import math
import gc
import warnings
from typing import cast

import numpy as np
import torch
import scipy.sparse as sp
from tqdm import tqdm
import psutil

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _ensure_device_tensor(value, device, dtype=torch.float32, require_dense=True):
    """Convert value to a torch tensor on the requested device.

    Args:
        value: torch.Tensor, numpy.ndarray, scipy sparse matrix, or scalar-like.
        device: Target torch device.
        dtype: Desired floating dtype. Ignored for boolean tensors.
        require_dense: If True, convert sparse tensors to dense representation.

    Returns:
        torch.Tensor on the requested device.
    """
    if torch.is_tensor(value):
        tensor = value.to(device)
        if dtype and tensor.dtype != dtype and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        if require_dense and (tensor.is_sparse or getattr(tensor, "is_sparse_csr", False)):
            tensor = tensor.to_dense()
        return tensor

    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
        tensor = tensor.to(device=device)
        if dtype and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        return tensor if not (require_dense and tensor.is_sparse) else tensor.to_dense()

    if sp.issparse(value):
        coo = value.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), device=device, dtype=torch.long)
        data_dtype = dtype if dtype is not None else torch.float32
        values = torch.tensor(coo.data, device=device, dtype=data_dtype)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size=coo.shape, device=device)
        return sparse_tensor.to_dense() if require_dense else sparse_tensor.coalesce()

    tensor = torch.tensor(value, device=device)
    if dtype and tensor.is_floating_point():
        tensor = tensor.to(dtype)
    return tensor


def _as_scalar_tensor(value, device, dtype=torch.float32):
    """Return a 0-dim tensor on device with dtype."""
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def _matvec(J_mat, x):
    """Matrix-vector product supporting dense or sparse J_mat."""
    if J_mat.is_sparse or getattr(J_mat, "is_sparse_csr", False) or getattr(J_mat, "is_sparse_csc", False):
        if J_mat.layout != torch.sparse_coo:
            J_mat = J_mat.to_sparse()
        return torch.sparse.mm(J_mat, x.unsqueeze(-1)).squeeze(-1)
    return J_mat @ x

class DOCH:
    """
    DOCH (Difference Of Convex Hamiltonian) solver for Ising model.
    """
    
    def __init__(self, device=None):
        """
        Initialize DOCH solver.
        
        Args:
            device: PyTorch device (cuda/cpu). Auto-detected if None.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def solve(self, J_mat, x0, eta, j_mat_2_norm, J_mat_1_norm, runtime):
        """Solve Ising model using DOCH.

        Args:
            J_mat: Coupling matrix.
            x0: Initial point.
            eta: Algorithm parameter.
            j_mat_2_norm: 2-norm of J.
            J_mat_1_norm: 1-norm of J.
            runtime: Maximum runtime.

        Returns:
            (energies, times, final_spins)
        """
        device = self.device
        J_mat = _ensure_device_tensor(J_mat, device=device, dtype=torch.float32, require_dense=False)
        x = _ensure_device_tensor(x0, device=device, dtype=torch.float32).flatten()

        eta_t = _as_scalar_tensor(eta, device=device)
        j_mat_2_t = _as_scalar_tensor(j_mat_2_norm, device=device)
        J_mat_1_t = _as_scalar_tensor(J_mat_1_norm, device=device)

        n = J_mat.shape[0]
        n_tensor = torch.tensor(n, device=device, dtype=torch.float32)
        
        # Step 1: Initialize parameters
        alpha = eta_t * j_mat_2_t
        beta = n_tensor * torch.sqrt(n_tensor) * (alpha + J_mat_1_t)
        
        energies = []
        times = []
        
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < runtime:
            # Step 2: DOCH update rule
            J_x = _matvec(J_mat, x)
            x = torch.sign(alpha * x + J_x) * (torch.abs((alpha * x + J_x) / beta))**(1/3)
            
            # Step 3: Compute energy and track progress
            spins = torch.sign(x)
            energy = -0.5 * (spins @ _matvec(J_mat, spins))
            
            energies.append(energy.item())
            times.append(time.time() - start_time)
            iterations += 1
            
            print(f'DOCH: {iterations} iter, {times[-1]:.3f}s, Energy: {energies[-1]:.3f}', end='\r')
        
        return energies, times, torch.sign(x)


class ADOCH:
    """
    ADOCH (Accelerated DOCH) solver for Ising model.
    """
    
    def __init__(self, device=None):
        """
        Initialize ADOCH solver.
        
        Args:
            device: PyTorch device (cuda/cpu). Auto-detected if None.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def solve(self, J_mat, x0, eta, j_mat_2_norm, J_mat_1_norm, runtime, q=10, T=5):
        """Solve Ising model using ADOCH.

        Args:
            J_mat: Coupling matrix.
            x0: Initial point.
            eta: Algorithm parameter.
            j_mat_2_norm: 2-norm of J.
            J_mat_1_norm: 1-norm of J.
            runtime: Maximum runtime.
            q: History window for adaptive restart.
            T: Number of DOCH iterations per step.

        Returns:
            (energies, times, final_spins)
        """
        device = self.device
        J_mat = _ensure_device_tensor(J_mat, device=device, dtype=torch.float32, require_dense=False)
        x = _ensure_device_tensor(x0, device=device, dtype=torch.float32).flatten()

        eta_t = _as_scalar_tensor(eta, device=device)
        j_mat_2_t = _as_scalar_tensor(j_mat_2_norm, device=device)
        J_mat_1_t = _as_scalar_tensor(J_mat_1_norm, device=device)

        n = J_mat.shape[0]
        n_tensor = torch.tensor(n, device=device, dtype=torch.float32)
        
        # Step 1: Initialize parameters
        alpha = eta_t * j_mat_2_t
        beta = n_tensor * torch.sqrt(n_tensor) * (alpha + J_mat_1_t)
        inv_beta = 1.0 / beta

        z = x.clone()
        t_val = 1.0
        
        energies = []
        times = []
        x_history = []
        
        def compute_objective(x_input):
            """Objective function: G(x) - H(x)"""
            quadratic = 0.5 * x_input @ (alpha * x_input + _matvec(J_mat, x_input))
            quartic = 0.25 * beta * torch.sum(x_input**4)
            return quartic - quadratic
        
        start_time = time.time()
        k = 0
        
        while time.time() - start_time < runtime:
            # Step 2: Compute momentum coefficient
            t_next = (1 + torch.sqrt(torch.tensor(1 + 4 * t_val**2, 
                                               device=self.device, dtype=torch.float32))) / 2
            
            x_prev = x.clone()
            
            # Step 3: Momentum update
            if k > 0:
                z = x + ((t_val - 1) / t_next) * (x - x_prev)
            else:
                z = x.clone()
            
            # Step 4: Adaptive restart
            if k > q:
                F_z = compute_objective(z)
                start_idx = max(0, k - q)
                
                if start_idx < k:
                    F_values = torch.stack([compute_objective(x_history[i]) 
                                          for i in range(start_idx, k)])
                    F_max = torch.max(F_values)
                    v = z if F_z <= F_max else x
                else:
                    v = z
            else:
                v = x
            
            x = v.clone()
            
            # Step 5: T iterations of DOCH update
            for _ in range(T):
                numerator = alpha * x + _matvec(J_mat, x)
                x = torch.sign(numerator) * torch.abs(numerator * inv_beta) ** (1/3)
            
            x_history.append(x.clone())
            t_val = t_next.item()
            
            # Compute energy and track progress
            spins = torch.sign(x)
            energy = -0.5 * (spins @ _matvec(J_mat, spins))
            
            elapsed_time = time.time() - start_time
            energies.append(energy.item())
            times.append(elapsed_time)
            
            print(f'ADOCH: {k+1} iter, {elapsed_time:.3f}s, Energy: {energy.item():.3f}', end='\r')
            k += 1
        
        return energies, times, torch.sign(x)


class SA:
    """Simulated Annealing for the Ising model.

    solve(J_mat, x0, beta0, runtime) -> (energies, times, spins)
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def solve(self, J_mat, x0, beta0: float, runtime: float):
        device = self.device
        J_mat = _ensure_device_tensor(J_mat, device=device, dtype=torch.float32, require_dense=False)
        x0_t = _ensure_device_tensor(x0, device=device, dtype=torch.float32).flatten()
        beta0_t = _as_scalar_tensor(beta0, device=device)

        N = J_mat.shape[0]
        spin = torch.sign(x0_t)
        E = -0.5 * spin @ _matvec(J_mat, spin)

        E_list = [float(E.item())]
        T_list = [0.0]

        start = time.time()
        t = 0.0
        it = 0
        while t < runtime:
            elapsed = t
            # Log temperature schedule; 
            beta = beta0_t * torch.log(torch.tensor(1.0 + elapsed / max(runtime, 1e-8),
                                                  device=device, dtype=torch.float32))

            v = int(torch.randint(0, N, (1,), device=device, dtype=torch.long).item())
            spin_new = spin.clone()
            spin_new[v] *= -1

            delta_E = -0.5 * spin_new @ _matvec(J_mat, spin_new) - E
            # delta_E = 2.0 * spin_new[v] * (J_mat[v, :] @ spin_new)

            accept_prob = torch.exp(-beta * delta_E)
            if delta_E.item() < 0 or torch.rand(1, device=device, dtype=accept_prob.dtype) < accept_prob:
                spin = spin_new
                E = E + delta_E

            t = time.time() - start
            E_list.append(float(E.item()))
            T_list.append(t)
            it += 1
            print(f'SA: {it} iter, {t:.3f}s, Energy: {E_list[-1]:.3f}', end='\r')

        return E_list, T_list, spin


class BSB:
    """Ballistic spin/bifurcation machine dynamics for the Ising model.

    solve(J_mat, x0, a0, c0, dt, runtime) -> (energies, times, spins)
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def solve(self, J_mat, x0, a0: float, c0: float, dt: float, runtime: float):
        device = self.device
        J_mat = _ensure_device_tensor(J_mat, device=device, dtype=torch.float32, require_dense=False)
        x0_t = _ensure_device_tensor(x0, device=device, dtype=torch.float32).flatten()
        a0_t = _as_scalar_tensor(a0, device=device)
        c0_t = _as_scalar_tensor(c0, device=device)
        dt_t = _as_scalar_tensor(dt, device=device)

        N = J_mat.shape[0]
        x = torch.sign(x0_t).flatten()
        y = torch.zeros(N, device=device, dtype=torch.float32)

        E_list = []
        T_list = []

        start = time.time()
        t = 0.0
        it = 0
        while t < runtime:
            a_t = a0_t * (time.time() - start) / max(runtime, 1e-8)
            y = y + (-(a0_t - a_t) * x + c0_t * _matvec(J_mat, x)) * dt_t
            x = x + a0_t * y * dt_t
            x = torch.clamp(x, -1.0, 1.0)
            y = torch.where((x == 1.0) | (x == -1.0), torch.zeros_like(y), y)

            ss = torch.sign(x)
            energy = -0.5 * (ss @ _matvec(J_mat, ss))
            t = time.time() - start
            E_list.append(float(energy.item()))
            T_list.append(t)
            it += 1
            print(f'BSB: {it} iter, {t:.3f}s, Energy: {E_list[-1]:.3f}', end='\r')

        return E_list, T_list, ss


class SimCIM:
    """Simulated coherent Ising machine dynamics.

    solve(J_mat, x0, A, a0, c0, dt, runtime) -> (energies, times, spins)
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def solve(self, J_mat, x0, A: float, a0: float, c0: float, dt: float, runtime: float):
        device = self.device
        J_mat = _ensure_device_tensor(J_mat, device=device, dtype=torch.float32, require_dense=False)
        x0_t = _ensure_device_tensor(x0, device=device, dtype=torch.float32).flatten()
        A_t = _as_scalar_tensor(A, device=device)
        a0_t = _as_scalar_tensor(a0, device=device)
        c0_t = _as_scalar_tensor(c0, device=device)
        dt_t = _as_scalar_tensor(dt, device=device)

        x = torch.sign(x0_t).flatten()

        E_list = []
        T_list = []
        start = time.time()
        t = 0.0
        it = 0
        sqrt_dt = torch.sqrt(dt_t)

        while t < runtime:
            a_t = a0_t * (time.time() - start) / max(runtime, 1e-8)
            noise = A_t * torch.randn_like(x, device=device) * sqrt_dt
            spin = torch.sign(x)
            x = x + (-(a0_t - a_t) * x + c0_t * _matvec(J_mat, spin)) * dt_t + noise
            x = torch.clamp(x, -1.0, 1.0)
            ss = torch.sign(x)

            energy = -0.5 * (ss @ _matvec(J_mat, ss))
            t = time.time() - start
            E_list.append(float(energy.item()))
            T_list.append(t)
            it += 1
            print(f'SimCIM: {it} iter, {t:.3f}s, Energy: {E_list[-1]:.3f}', end='\r')

        return E_list, T_list, ss


class SIS:
    """Spring-damping-based Ising machine dynamics.

    solve(J_mat, x0, m, k, zeta0, delta_t, runtime) -> (energies, times, spins)
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def solve(self, J_mat, x0, m: float, k: float, zeta0: float, delta_t: float, runtime: float):
        device = self.device
        J_mat = _ensure_device_tensor(J_mat, device=device, dtype=torch.float32, require_dense=False)
        x0_t = _ensure_device_tensor(x0, device=device, dtype=torch.float32)
        m_t = _as_scalar_tensor(m, device=device)
        k_t = _as_scalar_tensor(k, device=device)
        zeta0_t = _as_scalar_tensor(zeta0, device=device)
        delta_t_t = _as_scalar_tensor(delta_t, device=device)

        N = J_mat.shape[0]
        q = torch.zeros(N, device=device, dtype=torch.float32)
        p = 0.0005 * x0_t

        current_zeta = 0.8 * zeta0_t
        zeta_growth_rate = (10.0 * zeta0_t - 0.8 * zeta0_t) / max(runtime, 1e-8)

        E_list = []
        T_list = []
        start = time.time()
        t = 0.0
        it = 0

        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device, dtype=torch.float32))
        while t < runtime:
            q = q + delta_t_t * p / m_t
            p = p - delta_t_t * k_t * q + 0.5 * current_zeta * delta_t_t * _matvec(J_mat, q)
            q = torch.clamp(q, -sqrt_2, sqrt_2)
            p = torch.clamp(p, -2.0, 2.0)

            spin = torch.sign(q)
            energy = -0.5 * (spin @ _matvec(J_mat, spin))
            t = time.time() - start
            E_list.append(float(energy.item()))
            T_list.append(t)
            it += 1
            print(f'SIS: {it} iter, {t:.3f}s, Energy: {E_list[-1]:.3f}', end='\r')

            current_zeta = 0.8 * zeta0_t + zeta_growth_rate * t

        return E_list, T_list, spin



# Utility functions for small Ising matrices

def generate_random_ising(model, n, p=100.0, device=None):
    """
    Generate random symmetric Ising coupling matrix.
    
    Args:
        n: Matrix size
        model: 'sk' (Sherrington-Kirkpatrick) or 'fc' (Fully-Connected)
        p: Connectivity percentage (0 < p <= 100), percentage of nonzero couplings
        device: PyTorch device
        
    Returns:
        torch.Tensor: Symmetric coupling matrix with zero diagonal
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model not in ['sk', 'fc']:
        raise ValueError("Ising model must be 'sk (Sherrington-Kirkpatrick)' or 'fc' (Fully-Connected), else directly load J matrix.")
    elif model == 'sk':
        J_mat = torch.randn(n, n, device=device, dtype=torch.float32)
    else:
        fc_upper = torch.triu(torch.randint(0, 2, (n, n), device=device, dtype=torch.int32))
        J_mat = 2 * (fc_upper.float() * 2 - 1)
    
    # Make symmetric
    J_mat = (J_mat + J_mat.T)/2
    # Zero diagonal
    J_mat = J_mat - torch.diag(J_mat.diagonal())
    
    # Apply connectivity constraint
    if p < 100.0 and model != 'fc':
        # Create mask for upper triangular part (excluding diagonal)
        mask = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
        upper_indices = torch.where(mask)
        
        # Total number of upper triangular elements
        total_elements = n * (n - 1) // 2
        # Number of elements to keep
        keep_elements = int((p/100) * total_elements)
        
        # Randomly select indices to keep
        perm = torch.randperm(total_elements, device=device)
        keep_mask = torch.zeros(total_elements, device=device, dtype=torch.bool)
        keep_mask[perm[:keep_elements]] = True
        
        # Create sparsity mask
        sparsity_mask = torch.zeros(n, n, device=device, dtype=torch.bool)
        sparsity_mask[upper_indices[0][keep_mask], upper_indices[1][keep_mask]] = True
        # Make symmetric
        sparsity_mask = sparsity_mask | sparsity_mask.T
        
        # Apply mask
        J_mat = J_mat * sparsity_mask.float()
    
    return J_mat

def compute_matrix_norms(J_mat, device=None):
    """
    Compute matrix norms needed for DOCH/ADOCH algorithms.
    
    Args:
        J_mat: Coupling matrix
        
    Returns:
        tuple: (matrix_1_norm, matrix_2_norm)
    """
    if device is None:
        device = J_mat.device if torch.is_tensor(J_mat) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = _ensure_device_tensor(J_mat, device=device, dtype=torch.float32)
    n = tensor.shape[0]
    
    # 1-norm (maximum absolute column sum)
    J_mat_1_norm = torch.max(torch.abs(tensor).sum(dim=0))
    
    # 2-norm calculation
    if n <= 1000:
        # Exact computation for small matrices
        j_mat_2_norm = torch.linalg.norm(tensor, ord=2)
    else:
        # Wigner semicircle law approximation for large matrices
        j_mat_2_norm = 2 * math.sqrt(n) * torch.sqrt(
            torch.sum(tensor**2)/(n*(n-1)) - (torch.sum(tensor)/(n*(n-1)))**2
        )
    
    return J_mat_1_norm, j_mat_2_norm

def compute_j_bar(J_mat: torch.Tensor, device=None) -> torch.Tensor:

    if device is None:
        device = J_mat.device if torch.is_tensor(J_mat) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    J_mat = _ensure_device_tensor(J_mat, device=device, dtype=torch.float32)
    n = J_mat.shape[0]
    n_t = torch.tensor(float(n), device=J_mat.device, dtype=torch.float32)
    denom = n_t * (n_t - 1.0)
    return torch.sqrt(torch.sum(J_mat.float()**2) / denom)



# Utility functions for large Ising matrices

def generate_sparse_matrix_chunk(start_row, end_row, n, sparsity, seed=None, max_memory_percent=80):
    """Generate a chunk of the sparse matrix directly in CSR format."""
    # Set random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed + start_row)
    
    # Calculate expected number of non-zero elements per row (excluding diagonal)
    nnz_per_row = max(1, math.ceil((1 - sparsity) * (n - 1)))
    
    # Arrays for CSR format
    indptr = [0]  # Row pointer array
    indices = []  # Column indices
    data = []     # Values
    
    # Process each row in this chunk with memory monitoring
    current_nnz = 0
    for i in range(start_row, end_row):
        # Check memory usage periodically
        if (i - start_row) % 100 == 0:
            mem_usage = psutil.virtual_memory().percent
            if mem_usage > max_memory_percent:
                print(f"\nWarning: Memory usage at {mem_usage}%. Reducing non-zeros for remaining rows.")
                # Dynamically reduce nnz_per_row if memory is getting tight
                nnz_per_row = max(1, nnz_per_row // 2)
        
        # Generate column indices for this row (excluding diagonal)
        if n <= 10**5:  # For smaller matrices, we can do this more directly
            cols = np.random.choice(
                np.concatenate([np.arange(0, i), np.arange(i+1, n)]),
                size=min(nnz_per_row, n-1),
                replace=False
            )
        else:
            # For large matrices, we need to be more memory-efficient in sampling
            # Strategy: Generate a small random subset and then sample from that
            sample_size = min(10*nnz_per_row, n-1)  # Sample pool size
            
            # Generate random indices without creating large arrays
            cols = set()
            attempts = 0
            max_attempts = sample_size * 10  # Limit attempts to prevent infinite loops
            
            while len(cols) < min(nnz_per_row, n-1) and attempts < max_attempts:
                col = np.random.randint(0, n)
                if col != i:  # Skip diagonal
                    cols.add(col)
                attempts += 1
            
            cols = np.array(list(cols))
        
        # Sort column indices (required for CSR format)
        cols.sort()
        
        # Generate random values (9-bit signed integers)
        vals = np.random.randint(-(2**9)+1, (2**9)-1, size=len(cols), dtype=np.int16)
        
        # Add to CSR arrays
        indices.extend(cols)
        data.extend(vals)
        current_nnz += len(cols)
        indptr.append(current_nnz)
        
        # print progress with row number and memory usage
        if (i - start_row) % 1000 == 0 and i > start_row:
            mem = psutil.virtual_memory()
            print(f"\rRow {i}/{end_row-1} - Memory: {mem.percent}% used, {mem.available/1e9:.1f}GB free",
                  end="", flush=True)
            
            # Force garbage collection periodically
            gc.collect()
    
    # Convert lists to arrays
    indptr_arr = np.array(indptr, dtype=np.int32)
    indices_arr = np.array(indices, dtype=np.int32)
    data_arr = np.array(data, dtype=np.int16)
    
    return indptr_arr, indices_arr, data_arr

def create_symmetric_from_half(upper_half, n):
    """Create a symmetric matrix from the upper triangular half."""
    print("Creating symmetric matrix from upper triangular half...")
    
    # Get the transpose (lower triangular part)
    lower_half = upper_half.transpose()
    
    # Add the two halves
    result = upper_half + lower_half
    
    return result

def generate_sparse_matrix(n=10**6, sparsity=0.99, num_chunks=32, seed=42, max_memory_percent=80,
                           as_torch=False, device=None, require_dense=False, dtype=torch.float32):
    """
    Generate a large sparse symmetric matrix with the specified properties using CSR format.

    Args:
        n: Dimension of the square matrix.
        sparsity: Desired sparsity level (fraction of zero entries).
        num_chunks: Number of row chunks to stream during generation.
        seed: Random seed for reproducibility.
        max_memory_percent: Max allowed system memory usage before adaptive sparsity kicks in.
        as_torch: When True, returns a torch sparse tensor on the chosen device.
        device: Optional torch.device for conversion when as_torch is True.
        require_dense: If True and as_torch, convert the sparse tensor to dense representation.
        dtype: Target floating dtype for the torch tensor conversion.

    Returns:
        scipy.sparse matrix or torch.Tensor depending on `as_torch` flag.
    """
    print(f"Generating {n}x{n} sparse matrix with {sparsity*100:.7f}% sparsity")
    print(f"Using CSR format for memory efficiency")
    
    # Calculate chunk sizes
    chunk_size = n // num_chunks
    chunk_size = max(1, chunk_size)  # Ensure chunk size is at least 1
    
    # Initialize empty arrays for CSR format
    all_indptr = [0]
    all_indices = []
    all_data = []
    
    # Track memory usage
    peak_memory = 0
    start_time = time.time()
    
    # Process chunks to build the upper triangular part
    for i in range(0, n, chunk_size):
        chunk_start = i
        chunk_end = min(i + chunk_size, n)
        
        print(f"\nProcessing chunk {chunk_start}-{chunk_end} ({chunk_end-chunk_start} rows)")
        
        # Generate this chunk in CSR format
        indptr, indices, data = generate_sparse_matrix_chunk(
            chunk_start, chunk_end, n, sparsity, 
            seed=seed+i if seed else None,
            max_memory_percent=max_memory_percent
        )
        
        # Adjust the indptr values to account for the current total
        if len(all_indices) > 0:
            last_nnz = all_indptr[-1]
            indptr = indptr[1:] + last_nnz  # Skip the first element (0) and add the offset
        else:
            indptr = indptr[1:]  # Skip just the first element
            
        # Append to the main arrays
        all_indptr.extend(indptr)
        all_indices.extend(indices)
        all_data.extend(data)
        
        # Clear chunk data to free memory
        indptr = indices = data = None
        gc.collect()
        
        # Log memory usage
        mem = psutil.virtual_memory()
        if mem.percent > peak_memory:
            peak_memory = mem.percent
        
        elapsed = time.time() - start_time
        remaining_chunks = (n - chunk_end) / chunk_size
        est_remaining = remaining_chunks * (elapsed / ((chunk_start + chunk_size) / chunk_size))
        
        print(f"Memory: {mem.percent}% used, {mem.available/1e9:.1f}GB free")
        print(f"Progress: {100*chunk_end/n:.1f}% done, ~{est_remaining/60:.1f} minutes remaining")
    
    # Create the upper triangular matrix
    print("Constructing final upper triangular matrix...")
    upper_half = sp.csr_matrix((all_data, all_indices, all_indptr), shape=(n, n), dtype=np.int16)
    
    # Clear the arrays to free memory
    all_indptr = all_indices = all_data = None
    gc.collect()
    
    # Create symmetric matrix
    matrix = create_symmetric_from_half(upper_half, n)
    
    # Report statistics
    print(f"Generated matrix with shape {matrix.shape} and {matrix.nnz} non-zero elements")
    print(f"Generation time: {time.time() - start_time:.2f} seconds")
    print(f"Peak memory usage: {peak_memory}%")
    
    if as_torch:
        matrix = scipy_sparse_to_torch(matrix, device=device, dtype=dtype, require_dense=require_dense)

    return matrix

def save_to_compressed_format(matrix, filename="sparse_matrix.npz"):
    """Save the sparse matrix in a compressed format."""
    print(f"Saving matrix to {filename}")
    start_time = time.time()
    sp.save_npz(filename, matrix, compressed=True)
    save_time = time.time() - start_time
    file_size = os.path.getsize(filename) / (1024**3)
    print(f"File size: {file_size:.2f} GB, Save time: {save_time:.2f} seconds")


def scipy_sparse_to_torch(matrix, device=None, dtype=torch.float32, require_dense=False):
    """Convert a SciPy sparse matrix to a torch tensor on the requested device.

    Args:
        matrix: SciPy sparse matrix instance.
        device: Optional torch.device. Defaults to CUDA if available.
        dtype: Target dtype for the values tensor.
        require_dense: When True, returns a dense tensor.

    Returns:
        torch.Tensor: Sparse or dense tensor on the selected device.
    """
    if not sp.issparse(matrix):
        raise TypeError("Expected a SciPy sparse matrix.")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _ensure_device_tensor(matrix, device=device, dtype=dtype, require_dense=require_dense)

def build_large_matrix(n, sparsity):
    # System information
    print(f"CPU Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"CPU Cores: {os.cpu_count()}")
    
    # Calculate memory requirements
    total_elements = n**2
    nonzero_elements = int(total_elements * (1 - sparsity))
    estimated_memory_csr = ((nonzero_elements * (4 + 2)) + (n + 1) * 4) / (1024**3)  # CSR format in GB
    
    print(f"Matrix size: {n} x {n} = {total_elements} elements")
    print(f"Non-zero elements: ~{nonzero_elements} ({(1-sparsity)*100:.10f}%)")
    print(f"Estimated memory (CSR format): ~{estimated_memory_csr:.2f} GB")
    
    # Determine optimal chunk size based on available memory
    available_memory = psutil.virtual_memory().available / (1024**3)  # Available memory in GB
    
    # Adaptive chunk sizing
    # We want each chunk to use no more than 5% of available memory
    max_chunk_memory = available_memory * 0.05  # GB
    row_memory = estimated_memory_csr / n  # GB per row
    chunk_size = int(max_chunk_memory / row_memory)
    
    # Ensure reasonable chunk size
    chunk_size = min(max(chunk_size, 1000), 10000)
    num_chunks = math.ceil(n / chunk_size)
    
    print(f"Available memory: {available_memory:.2f} GB")
    print(f"Using {num_chunks} chunks with ~{chunk_size} rows per chunk")
    
    # Set memory threshold for adaptive reduction
    max_memory_percent = 90
    print(f"Will reduce non-zeros if memory usage exceeds {max_memory_percent}%")
    
    # Generate the matrix
    start_time = time.time()
    matrix = generate_sparse_matrix(
        n=n, 
        sparsity=sparsity, 
        num_chunks=num_chunks, 
        seed=42, 
        max_memory_percent=max_memory_percent
    )
    generation_time = time.time() - start_time
    
    # Report memory usage of final matrix
    if sp.issparse(matrix):
        matrix_csr = cast(sp.csr_matrix, matrix)
        matrix_memory = (matrix_csr.data.nbytes + matrix_csr.indices.nbytes + matrix_csr.indptr.nbytes) / (1024**3)
    elif isinstance(matrix, torch.Tensor):
        tensor_matrix = matrix
        is_sparse = tensor_matrix.is_sparse or getattr(tensor_matrix, "is_sparse_csr", False) or getattr(tensor_matrix, "is_sparse_csc", False)
        if is_sparse:
            if tensor_matrix.layout == torch.sparse_coo:
                storage = tensor_matrix.coalesce()
            else:
                storage = tensor_matrix.to_sparse().coalesce()
            matrix_memory = (
                storage.values().element_size() * storage.values().numel() +
                storage.indices().element_size() * storage.indices().numel()
            ) / (1024**3)
        else:
            matrix_memory = tensor_matrix.element_size() * tensor_matrix.numel() / (1024**3)
    else:
        matrix_memory = float("nan")
    print(f"Matrix memory usage: {matrix_memory:.2f} GB")
    print(f"Total generation time: {generation_time:.2f} seconds")
    
    # # Save the matrix
    # if not test_mode:
    #     save_to_compressed_format(matrix, filename="sparse_M6.npz")
    
    print("Process completed successfully")
    return matrix

def compute_J_bar(J_mat: torch.Tensor, device=None) -> torch.Tensor:
    """Compute j_bar = sqrt(sum(J^2)/(n*(n-1))). Useful for BSB/SimCIM parameterization.

    Args:
        J_mat: Coupling matrix (n x n) with zero diagonal preferred.

    Returns:
        torch.Tensor: scalar tensor j_bar on J_mat.device with dtype float32.
    """
    if device is None:
        device = J_mat.device if torch.is_tensor(J_mat) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    J_mat = _ensure_device_tensor(J_mat, device=device, dtype=torch.float32)
    n = J_mat.shape[0]
    n_t = torch.tensor(float(n), device=J_mat.device, dtype=torch.float32)
    denom = n_t * (n_t - 1.0) * (10 - 1.0)**2
    return torch.sqrt(torch.sum(J_mat.float()**2) / denom)

def calculate_parameters_with_progress(J_mat_coo, n=1000000, memory_efficient=True, chunk_size=50000):
    """
    Wrapper function that calculates parameters with progress bars
    
    Args:
        J_mat_coo: scipy.sparse.coo_matrix - The sparse matrix in COO format
        n: int - Matrix dimension (assuming square matrix)
        memory_efficient: bool - Whether to use the memory-efficient approach
        chunk_size: int - Size of chunks to process at once (for memory-efficient approach)
    
    Returns:
        dict: Dictionary containing the calculated parameters
    """
    print(f"Processing {J_mat_coo.nnz} non-zero elements in {n}x{n} matrix")
    
    if memory_efficient:
        print("Using memory-efficient approach")
        # Calculate j_mat_2_norm
        print("Calculating j_mat_2_norm...")
        squared_sum = 0
        data_chunks = np.array_split(J_mat_coo.data, max(1, len(J_mat_coo.data) // 10**6))
        for chunk in tqdm(data_chunks, desc="Summing squares", unit="chunk"):
            squared_sum += np.sum(chunk.astype(np.float32) ** 2)
        j_mat_2_norm = 2 * np.sqrt(n) * np.sqrt(squared_sum / (n * (n - 1)))
        
        # Calculate J_mat_1_norm
        print("Converting to CSC format for column operations...")
        J_mat_csc = J_mat_coo.tocsc()
        
        print("Calculating J_mat_1_norm...")
        max_col_sum = 0
        
        for start_col in tqdm(range(0, n, chunk_size), desc="Processing columns", unit="chunk"):
            end_col = min(start_col + chunk_size, n)
            
            # Extract a subset of columns
            J_sub = J_mat_csc[:, start_col:end_col]
            
            # Calculate column sums for this chunk
            col_sums = np.array(abs(J_sub).sum(axis=0)).flatten()
            
            # Update max column sum - FIX: Check if col_sums is not empty
            if col_sums.size > 0:
                chunk_max = np.max(col_sums)
                max_col_sum = max(max_col_sum, chunk_max)
        
        # Calculate j_bar
        print("Calculating j_bar in chunks...")
        j_bar = j_mat_2_norm / (2 * np.sqrt(n) * (10 - 1.0))
        
        return {
            "j_bar": j_bar,
            "J_mat_1_norm": max_col_sum, 
            "j_mat_2_norm": j_mat_2_norm
        }
    else:
        print("Using PyTorch-based approach")
        # Convert scipy COO matrix to PyTorch sparse tensor
        print("Converting to PyTorch sparse tensor...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        indices = torch.tensor(np.vstack((J_mat_coo.row, J_mat_coo.col)), dtype=torch.long, device=device)
        
        # Process values in chunks to avoid memory issues
        values_chunks = np.array_split(J_mat_coo.data, max(1, len(J_mat_coo.data) // 10000000))
        values = []

        for chunk in tqdm(values_chunks, desc="Processing values", unit="chunk"):
            values.append(torch.tensor(chunk, dtype=torch.float32, device=device))

        values = torch.cat(values)
        
        # Create PyTorch sparse tensor
        print(f"Using device: {device}")
        J_mat = torch.sparse_coo_tensor(indices, values, size=(n, n), device=device)
        
        # Calculate j_mat_2_norm
        print("Calculating j_mat_2_norm...")
        J_mat_coalesced = J_mat.coalesce()
        squared_sum = torch.sum(J_mat_coalesced.values().float() ** 2).item()
        j_mat_2_norm = 2 * np.sqrt(n) * np.sqrt(squared_sum / (n * (n - 1)))
        
        # Calculate J_mat_1_norm
        print("Calculating J_mat_1_norm...")
        J_abs = J_mat.abs()
        print("Computing column sums...")
        col_sums = torch.sparse.sum(J_abs, dim=0).to_dense()
        J_mat_1_norm = torch.max(col_sums).item()
        print(f"J_mat_1_norm = {J_mat_1_norm}")

        # Calculate j_bar
        print("Calculating j_bar...")
        j_bar = j_mat_2_norm / (2 * np.sqrt(n) * (10 - 1.0))
        
        return {
            "j_bar": j_bar,
            "J_mat_1_norm": J_mat_1_norm,
            "j_mat_2_norm": j_mat_2_norm
        }


