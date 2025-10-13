"""
Ising Model Solvers: DOCH, ADOCH, SA, BSB, SimCIM, SIS

This module implements multiple solvers for the Ising model:
    - DOCH (Difference Of Convex Hamiltonian)
    - ADOCH (Accelerated DOCH)
    - SA (Simulated Annealing)
    - BSB (Ballistic Spin/Bifurcation machine)
    - SimCIM (Simulated Coherent Ising Machine)
    - SIS (Spring-damping-based Ising Machine)

All solvers expose a solve(...) method returning (energies, times, final_spins).
"""

# import necessary libraries
import torch
import time
import math


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
        n = J_mat.shape[0]
        n_tensor = torch.tensor(n, device=self.device, dtype=torch.float32)
        
        # Step 1: Initialize parameters
        alpha = eta * j_mat_2_norm
        beta = n_tensor * torch.sqrt(n_tensor) * (alpha + J_mat_1_norm)
        
        energies = []
        times = []
        x = x0.clone()
        
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < runtime:
            # Step 2: DOCH update rule
            J_x = J_mat @ x
            x = torch.sign(alpha * x + J_x) * (torch.abs((alpha * x + J_x) / beta))**(1/3)
            
            # Step 3: Compute energy and track progress
            spins = torch.sign(x)
            energy = -0.5 * (spins @ J_mat @ spins)
            
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
        n = J_mat.shape[0]
        n_tensor = torch.tensor(n, device=self.device, dtype=torch.float32)
        
        # Step 1: Initialize parameters
        alpha = eta * j_mat_2_norm
        beta = n_tensor * torch.sqrt(n_tensor) * (alpha + J_mat_1_norm)
        inv_beta = 1.0 / beta
        
        x = x0.clone()
        z = x0.clone()
        t_val = 1.0
        
        energies = []
        times = []
        x_history = []
        
        def compute_objective(x_input):
            """Objective function: G(x) - H(x)"""
            quadratic = 0.5 * x_input @ (alpha * x_input + (J_mat @ x_input))
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
                numerator = alpha * x + (J_mat @ x)
                x = torch.sign(numerator) * torch.abs(numerator * inv_beta) ** (1/3)
            
            x_history.append(x.clone())
            t_val = t_next.item()
            
            # Compute energy and track progress
            spins = torch.sign(x)
            energy = -0.5 * (spins @ J_mat @ spins)
            
            elapsed_time = time.time() - start_time
            energies.append(energy.item())
            times.append(elapsed_time)
            
            print(f'ADOCH: {k+1} iter, {elapsed_time:.3f}s, Energy: {energy.item():.3f}', end='\r')
            k += 1
        
        return energies, times, torch.sign(x)



def compute_matrix_norms(J_mat):
    """
    Compute matrix norms needed for DOCH/ADOCH algorithms.
    
    Args:
        J_mat: Coupling matrix
        
    Returns:
        tuple: (matrix_1_norm, matrix_2_norm)
    """
    n = J_mat.shape[0]
    
    # 1-norm (maximum absolute column sum)
    J_mat_1_norm = torch.max(torch.abs(J_mat).sum(dim=0))
    
    # 2-norm calculation
    if n <= 1000:
        # Exact computation for small matrices
        j_mat_2_norm = torch.linalg.norm(J_mat, ord=2)
    else:
        # Wigner semicircle law approximation for large matrices
        j_mat_2_norm = 2 * math.sqrt(n) * torch.sqrt(
            torch.sum(J_mat**2)/(n*(n-1)) - (torch.sum(J_mat)/(n*(n-1)))**2
        )
    
    return J_mat_1_norm, j_mat_2_norm


def compute_J_bar(J_mat: torch.Tensor) -> torch.Tensor:
    """Compute j_bar = sqrt(sum(J^2)/(n*(n-1))). Useful for BSB/SimCIM parameterization.

    Args:
        J_mat: Coupling matrix (n x n) with zero diagonal preferred.

    Returns:
        torch.Tensor: scalar tensor j_bar on J_mat.device with dtype float32.
    """
    n = J_mat.shape[0]
    n_t = torch.tensor(float(n), device=J_mat.device, dtype=torch.float32)
    denom = n_t * (n_t - 1.0) * (10 - 1.0)**2
    return torch.sqrt(torch.sum(J_mat.float()**2) / denom)



class SA:
    """Simulated Annealing for the Ising model.

    solve(J_mat, x0, beta0, runtime) -> (energies, times, spins)
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def solve(self, J_mat, x0, beta0: float, runtime: float):
        N = J_mat.shape[0]
        device = self.device

        spin = torch.sign(x0.to(device))
        E = -0.5 * spin @ J_mat.to(device) @ spin

        E_list = [float(E.item())]
        T_list = [0.0]

        start = time.time()
        t = 0.0
        it = 0
        while t < runtime:
            elapsed = t
            # Log temperature schedule; 
            beta = beta0 * torch.log(torch.tensor(1.0 + elapsed / max(runtime, 1e-8),
                                                  device=device, dtype=torch.float32))

            v = torch.randint(0, N, (1,), device=device).item()
            spin_new = spin.clone()
            spin_new[v] *= -1

            delta_E = -0.5 * spin_new @ J_mat.to(device) @ spin_new - E
            # delta_E = 2.0 * spin_new[v] * (J_mat[v, :] @ spin_new)
            
            if delta_E < 0 or torch.rand(1, device=device, dtype=torch.float32).item() < torch.exp(-beta * delta_E).item():
                spin = spin_new
                E = E + delta_E

            t = time.time() - start
            E_list.append(float(E.item()))
            T_list.append(t)
            it += 1
            # if it % 1000 == 0:
            #     print(f'SA: {it} iter, {t:.3f}s, Energy: {E_list[-1]:.3f}', end='\r')

        return E_list, T_list, spin


class BSB:
    """Ballistic spin/bifurcation machine dynamics for the Ising model.

    solve(J_mat, x0, a0, c0, dt, runtime) -> (energies, times, spins)
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def solve(self, J_mat, x0, a0: float, c0: float, dt: float, runtime: float):
        N = J_mat.shape[0]
        device = self.device
        x = torch.sign(x0.to(device)).flatten()
        y = torch.zeros(N, device=device, dtype=torch.float32)

        E_list = []
        T_list = []

        start = time.time()
        t = 0.0
        it = 0
        while t < runtime:
            a_t = a0 * (time.time() - start) / max(runtime, 1e-8)
            y = y + (-(a0 - a_t) * x + c0 * (J_mat @ x)) * dt
            x = x + a0 * y * dt
            x = torch.clamp(x, -1.0, 1.0)
            y = torch.where((x == 1.0) | (x == -1.0), torch.zeros_like(y), y)

            ss = torch.sign(x)
            energy = -0.5 * (ss @ J_mat @ ss)
            t = time.time() - start
            E_list.append(float(energy.item()))
            T_list.append(t)
            it += 1
            # if it % 1000 == 0:
            #     print(f'BSB: {it} iter, {t:.3f}s, Energy: {E_list[-1]:.3f}', end='\r')

        return E_list, T_list, ss


class SimCIM:
    """Simulated coherent Ising machine dynamics.

    solve(J_mat, x0, A, a0, c0, dt, runtime) -> (energies, times, spins)
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def solve(self, J_mat, x0, A: float, a0: float, c0: float, dt: float, runtime: float):
        device = self.device
        x = torch.sign(x0.to(device)).flatten()

        E_list = []
        T_list = []
        start = time.time()
        t = 0.0
        it = 0
        sqrt_dt = torch.sqrt(torch.tensor(dt, device=device, dtype=torch.float32))

        while t < runtime:
            a_t = a0 * (time.time() - start) / max(runtime, 1e-8)
            noise = A * torch.randn_like(x, device=device) * sqrt_dt
            x = x + (-(a0 - a_t) * x + c0 * (J_mat @ torch.sign(x))) * dt + noise
            x = torch.clamp(x, -1.0, 1.0)
            ss = torch.sign(x)

            energy = -0.5 * (ss @ J_mat @ ss)
            t = time.time() - start
            E_list.append(float(energy.item()))
            T_list.append(t)
            it += 1
            # if it % 1000 == 0:
            #     print(f'SimCIM: {it} iter, {t:.3f}s, Energy: {E_list[-1]:.3f}', end='\r')

        return E_list, T_list, ss


class SIS:
    """Spring-damping-based Ising machine dynamics.

    solve(J_mat, x0, m, k, zeta0, delta_t, runtime) -> (energies, times, spins)
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def solve(self, J_mat, x0, m: float, k: float, zeta0: float, delta_t: float, runtime: float):
        device = self.device
        N = J_mat.shape[0]
        q = torch.zeros(N, device=device, dtype=torch.float32)
        p = 0.0005 * x0.to(device)

        current_zeta = 0.8 * zeta0
        zeta_growth_rate = (10.0 * zeta0 - 0.8 * zeta0) / max(runtime, 1e-8)

        E_list = []
        T_list = []
        start = time.time()
        t = 0.0
        it = 0

        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device, dtype=torch.float32))
        while t < runtime:
            q = q + delta_t * p / m
            p = p - delta_t * k * q + 0.5 * current_zeta * delta_t * (J_mat @ q)
            q = torch.clamp(q, -sqrt_2, sqrt_2)
            p = torch.clamp(p, -2.0, 2.0)

            spin = torch.sign(q)
            energy = -0.5 * (spin @ J_mat @ spin)
            t = time.time() - start
            E_list.append(float(energy.item()))
            T_list.append(t)
            it += 1
            # if it % 1000 == 0:
            #     print(f'SIS: {it} iter, {t:.3f}s, Energy: {E_list[-1]:.3f}', end='\r')

            current_zeta = 0.8 * zeta0 + zeta_growth_rate * t

        return E_list, T_list, spin



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
        J_mat = 2*torch.triu(torch.randint(0, 2, (n, n), device=device, dtype=torch.float32) * 2 - 1)
    
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




def compute_matrix_norms(J_mat):
    """
    Compute matrix norms needed for DOCH/ADOCH algorithms.
    
    Args:
        J_mat: Coupling matrix
        
    Returns:
        tuple: (matrix_1_norm, matrix_2_norm)
    """
    n = J_mat.shape[0]
    
    # 1-norm (maximum absolute column sum)
    J_mat_1_norm = torch.max(torch.abs(J_mat).sum(dim=0))
    
    # 2-norm calculation
    if n <= 1000:
        # Exact computation for small matrices
        j_mat_2_norm = torch.linalg.norm(J_mat, ord=2)
    else:
        # Wigner semicircle law approximation for large matrices
        j_mat_2_norm = 2 * math.sqrt(n) * torch.sqrt(
            torch.sum(J_mat**2)/(n*(n-1)) - (torch.sum(J_mat)/(n*(n-1)))**2
        )
    
    return J_mat_1_norm, j_mat_2_norm


def compute_j_bar(J_mat: torch.Tensor) -> torch.Tensor:
    """Compute j_bar = sqrt(sum(J^2)/(n*(n-1))). Useful for BSB/SimCIM parameterization.

    Args:
        J_mat: Coupling matrix (n x n) with zero diagonal preferred.

    Returns:
        torch.Tensor: scalar tensor j_bar on J_mat.device with dtype float32.
    """
    n = J_mat.shape[0]
    n_t = torch.tensor(float(n), device=J_mat.device, dtype=torch.float32)
    denom = n_t * (n_t - 1.0)
    return torch.sqrt(torch.sum(J_mat.float()**2) / denom)

