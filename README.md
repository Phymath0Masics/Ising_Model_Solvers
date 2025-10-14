# Ising Model Solvers: DOCH, ADOCH, SA, BSB, SimCIM, SIS

This project provides multiple algorithms to solve the Ising model by approximating the lowest energy state:
- DOCH — Difference Of Convex Hamiltonian based Ising Solver
- ADOCH — Accelerated DOCH
- SA — Simulated Annealing
- BSB — Ballistic Simulated Bifurcation machine
- SimCIM — Simulated Coherent Ising Machine
- SIS — Spring-damping-based Ising machine
- Utility functions for generating small and large random Ising matrices and computing norms.


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
# Ising Model Solvers: DOCH, ADOCH, SA, BSB, SimCIM, SIS

This project provides multiple algorithms to solve the Ising model by approximating the lowest energy state:
- DOCH — Difference Of Convex Hamiltonian based Ising Solver
- ADOCH — Accelerated DOCH
- SA — Simulated Annealing
- BSB — Ballistic (Simulated) Bifurcation machine
- SimCIM — Simulated Coherent Ising Machine
- SIS — Spring-damping-based Ising machine
- Utilities to generate small/large Ising matrices and compute derived parameters/norms.


## What’s inside
- `Ising_Solvers.py`: All solver classes and utilities (including large sparse matrix builders using SciPy and psutil)
- `demo_run.ipynb`: End-to-end benchmark notebook that:
  - builds a small dense binarized symmetric Ising matrix (Step 2.1)
  - optionally builds a very large sparse matrix (Step 2.2)
  - tunes DOCH and ADOCH (Steps 5.x, 6.x)
  - tunes and runs SA, BSB, SimCIM, SIS (Steps 7.x)
  - compares convergence and plots results (Steps 8–9)
- `environment.yml`: Conda environment spec


## Setup (Windows + Conda)
1) Install Miniconda if needed: https://docs.conda.io/en/latest/miniconda.html
2) Open a terminal in this folder.
3) Create and activate the environment:

```pwsh
conda env create -f environment.yml
conda activate ising-solvers
```

Notes:
- CPU works out of the box. For NVIDIA GPUs, see “GPU acceleration (optional)” below.
- The environment includes: numpy, torch, scipy, matplotlib, jupyter, tqdm, psutil.


## Run the demo
Open `demo_run.ipynb` and Run All. The notebook is structured with numbered steps and prints progress and summary lines.

Two paths are offered in Step 2:
- Small dense matrix (default in 2.1): quick to try; uses PyTorch tensors
- Large sparse matrix (2.2): builds an n up to 1e5 example using SciPy CSR; uses psutil to monitor memory; then computes norms/parameters chunk-wise

Tip: If you only need the small case, skip Step 2.2 cells to save time and memory.


## Minimal code example (small dense case)
```python
from Ising_Solvers import (
    DOCH, ADOCH, SA, BSB, SimCIM, SIS,
    compute_matrix_norms, generate_random_ising, compute_j_bar
)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n = 1000
J_mat = generate_random_ising('sk', n, p=50.0, device=device)
J_mat_1_norm, j_mat_2_norm = compute_matrix_norms(J_mat)
x0 = torch.randn(n, device=device)
j_bar = compute_j_bar(J_mat)

doch = DOCH(device)
adoch = ADOCH(device)
E_DOCH, T_DOCH, s_DOCH = doch.solve(J_mat, x0, eta=1.0, j_mat_2_norm=j_mat_2_norm, J_mat_1_norm=J_mat_1_norm, runtime=5)
E_ADOCH, T_ADOCH, s_ADOCH = adoch.solve(J_mat, x0, eta=0.1, j_mat_2_norm=j_mat_2_norm, J_mat_1_norm=J_mat_1_norm, runtime=5)

# Optional others
sa = SA(device)
bsb = BSB(device)
cim = SimCIM(device)
sis = SIS(device)
c0 = float(4.5/(j_bar*torch.sqrt(torch.tensor(float(n), device=J_mat.device))))
E_SA, T_SA, s_SA = sa.solve(J_mat, x0, beta0=1.0, runtime=5)
E_BSB, T_BSB, s_BSB = bsb.solve(J_mat, x0, a0=1.0, c0=c0, dt=1e-2, runtime=5)
E_CIM, T_CIM, s_CIM = cim.solve(J_mat, x0, A=0.1, a0=1.0, c0=c0, dt=1e-2, runtime=5)
E_SIS, T_SIS, s_SIS = sis.solve(J_mat, x0, m=1.0, k=0.5, zeta0=0.05, delta_t=2e-1, runtime=5)
```


## Large sparse case notes (Step 2.2)
- Uses SciPy CSR matrices and chunked operations for norms/parameters.
- Requires significant RAM when n is large; the notebook adapts non-zeros if memory pressure is detected.
- Outputs j_bar, J_mat_1_norm, and j_mat_2_norm via `calculate_parameters_with_progress`.


## GPU acceleration (optional)
If you have an NVIDIA GPU, you can create the env with CUDA support. One option is to add this line in `environment.yml` (under dependencies) and recreate the env:

```
pytorch-cuda=12.1
```

Ensure the channels include `nvidia` and that your PyTorch build matches the CUDA version. Otherwise the code will transparently run on CPU.


## Troubleshooting
- Conda env: Ensure it’s activated: `conda activate ising-solvers`.
- Notebook kernel: Select the `ising-solvers` Python kernel in Jupyter.
- Out of memory in Step 2.2: Reduce `n`, increase `sparsity`, or skip the large-matrix cells.
- CUDA not found: It will fall back to CPU. To use GPU, install a CUDA-capable PyTorch and drivers.


## License
This repository is provided as-is for research and educational use. See header comments for implementation details.
