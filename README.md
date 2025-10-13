# Ising Model Solvers: DOCH, ADOCH, SA, BSB, SimCIM, SIS

This project provides multiple algorithms to solve the Ising model by approximating the lowest energy state:
- DOCH — Difference Of Convex Hamiltonian based Ising Solver
- ADOCH — Accelerated DOCH
- SA — Simulated Annealing
- BSB — Ballistic Simulated Bifurcation machine
- SimCIM — Simulated Coherent Ising Machine
- SIS — Spring-damping-based Ising machine


## 🎯 What it Does
- Solves Ising problems using six different algorithms (listed above)
- Includes easy-to-use Python API and a demo notebook.


## 📋 Prerequisites
No programming experience needed to run the demo.


## Run on Google Colab
You can run the demo directly from the Jupyter notebook file `demo_run.ipynb`.




## 📁 Key Project Files
```
ising-solvers/
├── Ising_Solvers.py          # Algorithm classes (DOCH, ADOCH, SA, BSB, SimCIM, SIS)
├── demo_run.ipynb            # Interactive Jupyter notebook demo
├── environment.yml           # Conda environment specification
├── README.md                 # This file
```


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🚀 Quick Setup to run on device
1. **Install Conda**: If you don't have it, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Get Project**: Download or clone this repository.

3. **Set up Environment**:
   * Open a terminal/command prompt in the project folder (e.g., `cd path/to/ising-solvers`).
   * Create the environment: `conda env create -f environment.yml`
   * Activate the environment: `conda activate ising-solvers`
   
4. **Run Demo**:
   * **Jupyter Notebook**: open `demo_run.ipynb` and run all cells
   * **Google Colab**: Upload `demo_run.ipynb` and click 'Run All'



## 🔧 How to Use (Basic Example)

```python
from Ising_Solvers import (
   DOCH, ADOCH, SA, BSB, SimCIM, SIS,
   compute_matrix_norms, generate_random_ising, compute_j_bar
)
import torch

# 1. Setup problem
n = 1000  # Size of the problem
p = 50.0  # Connectivity percentage (0 to 100%), only for model = 'sk'
model = 'sk'  # 'sk' for Sherrington-Kirkpatrick Ising model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
J_mat = generate_random_ising(model, n, p, device) # Ising matrix
J_mat_1_norm, j_mat_2_norm = compute_matrix_norms(J_mat)
x0 = torch.randn(n, device=device) # Initial guess
j_bar = compute_j_bar(J_mat)

# 2. Create solvers
doch_solver = DOCH(device)
adoch_solver = ADOCH(device)
sa_solver = SA(device)
bsb_solver = BSB(device)
simcim_solver = SimCIM(device)
sis_solver = SIS(device)

# 3. Solve
eta = 1.0  # Algorithm parameter
runtime = 10.0  # Max time in seconds
energies_doch, _, spins_doch = doch_solver.solve(J_mat, x0, eta, j_mat_2_norm, J_mat_1_norm, runtime)
energies_adoch, _, spins_adoch = adoch_solver.solve(J_mat, x0, eta, j_mat_2_norm, J_mat_1_norm, runtime)

# Additional solvers (simple defaults)
energies_sa, _, spins_sa = sa_solver.solve(J_mat, x0, beta0=1.0, runtime=runtime)
c0 = float(4.5/(j_bar*torch.sqrt(torch.tensor(float(n), device=J_mat.device))))
energies_bsb, _, spins_bsb = bsb_solver.solve(J_mat, x0, a0=1.0, c0=c0, dt=1e-2, runtime=runtime)
energies_cim, _, spins_cim = simcim_solver.solve(J_mat, x0, A=0.1, a0=1.0, c0=c0, dt=1e-2, runtime=runtime)
energies_sis, _, spins_sis = sis_solver.solve(J_mat, x0, m=1.0, k=0.5, zeta0=0.05, delta_t=2e-1, runtime=runtime)

# 'energies' lists energy values, 'spins' is the final solution spin vector ({±1}^n)
```

## 📊 Demo Output
When you run the demo, you'll see:
- Information about the problem being solved.
- Progress of the DOCH and ADOCH algorithms.
- Final energy values and a plot comparing their performance.
```
# Example output snippet:
DOCH: ... Energy: -1234.567
ADOCH: ... Energy: -1345.678
```

## 🎯 Algorithms Briefly
- **DOCH**: Solves Ising problems using a continuous dynamics approach.
- **ADOCH**: An accelerated version of DOCH, often faster and more effective for larger problems.
- **SA**: Simulated annealing using Metropolis updates on spins.
- **BSB**: Ballistic dynamics with saturation and coupling feedback.
- **SimCIM**: Stochastic continuous dynamics approximating coherent Ising machines.
- **SIS**: Second-order spring-damper style dynamics driven by the couplings.

## 🐛 Troubleshooting
- **Environment Issues**: Make sure Conda is installed and the `ising-solvers` environment is activated.
- **CUDA Errors**: The code automatically uses CPU if a GPU (CUDA) isn't available.
- **Memory/Speed**: For large problems, try reducing the size (`n`) or ensure you're using CPU if GPU memory is an issue.

## 🏆 Expected Results
- ADOCH generally converges faster (requires less number of iterations) and can find better (lower) energy solutions than DOCH, especially for larger problems.
- Both algorithms aim to find high-quality solutions.
