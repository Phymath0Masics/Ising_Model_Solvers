# Ising Model Solvers: DOCH and ADOCH

This project provides two algorithms, **DOCH** and **ADOCH**, to solve the Ising model by approximating the lowest energy state.


## 🎯 What it Does
- Solves Ising problems using:
   - **DOCH**: Difference Of Convex Hamiltonian based Ising Solver
   - **ADOCH**: Acclerated Difference Of Convex Hamiltonian based Ising Solver
- Includes easy-to-use Python code and a demo.


## 📋 Prerequisites
No programming experience needed to run the demo.


## Run on Google Colab
You can run the demo directly on [Google Colab](https://colab.research.google.com/github/yourusername/ising-solvers/blob/main/run_code.ipynb). 

Just upload or open the `run_code.ipynb` file to Google Colab, then click 'Run All', and follow the outputs.




## 📁 Key Project Files
```
ising-solvers/
├── Ising_Solvers.py          # Main algorithm classes (DOCH, ADOCH)
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
   * **Jupyter Notebook**: `jupyter notebook run_code.ipynb`
   * **Python Script**: `python run_code.py`
   * **Google Colab (Easiest)**: Upload `run_code.ipynb` to [Google Colab](https://colab.research.google.com), click 'Run All', and follow the outputs.



## 🔧 How to Use (Basic Example)

```python
from Ising_Solvers import DOCH, ADOCH, compute_matrix_norms, generate_random_ising
import torch

# 1. Setup problem
n = 1000  # Size of the problem
p = 50.0  # Connectivity percentage (0 to 100%), only for model = 'sk'
model = 'sk'  # 'sk' for Sherrington-Kirkpatrick Ising model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
J_mat = generate_random_ising(model, n, p, device) # Ising matrix
J_mat_1_norm, j_mat_2_norm = compute_matrix_norms(J_mat)
x0 = torch.randn(n, device=device) # Initial guess

# 2. Create solvers
doch_solver = DOCH(device)
adoch_solver = ADOCH(device)

# 3. Solve
eta = 1.0  # Algorithm parameter
runtime = 10.0  # Max time in seconds
energies_doch, _, spins_doch = doch_solver.solve(J_mat, x0, eta, j_mat_2_norm, J_mat_1_norm, runtime)
energies_adoch, _, spins_adoch = adoch_solver.solve(J_mat, x0, eta, j_mat_2_norm, J_mat_1_norm, runtime)

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

## 🐛 Troubleshooting
- **Environment Issues**: Make sure Conda is installed and the `ising-solvers` environment is activated.
- **CUDA Errors**: The code automatically uses CPU if a GPU (CUDA) isn't available.
- **Memory/Speed**: For large problems, try reducing the size (`n`) or ensure you're using CPU if GPU memory is an issue.

## 🏆 Expected Results
- ADOCH generally converges faster (requires less number of iterations) and can find better (lower) energy solutions than DOCH, especially for larger problems.
- Both algorithms aim to find high-quality solutions.
