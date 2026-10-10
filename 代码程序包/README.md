# Smart Healthcare Profit Model

Deterministic numerical experiments for a study of the boundary conditions
under which AI changes a smart healthcare platform's profit.

Authors: 张戈、卞慧馨、李佳路 (2026).
Repository: https://github.com/BHXHHHH/smart-healthcare-profit-model

## Installation

Reference environment: Python 3.9.0, NumPy 2.0.2, Matplotlib 3.9.4.
Create a separate environment; do not upload the environment directory.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.
Alternatively, run `conda env create -f environment.yml`, followed by
`conda activate smart-healthcare-profit`.

Install Times New Roman locally before generating figures. The plotting stage
checks its availability and fails explicitly if it is missing, rather than
silently substituting another font. Font files are not distributed here.

## Reproduction

No real patient data, downloaded datasets, or machine-learning training are
required. Inputs are theoretical parameters defined in the code.
The main experiment enumerates 10,368 parameter combinations.
Run from the project directory:

```powershell
python reproduce.py all
```

This runs the core experiment, analyses, figures, and reference-result checks.
Individual stages are also available:

```powershell
python reproduce.py core
python reproduce.py analysis
python reproduce.py figures
python reproduce.py verify
python -m unittest discover -s tests -v
```

`reproduce_single.py` is the complete single-file implementation;
`python reproduce_single.py all` is equivalent to the main entry point.
Tables and figures are written to `output/` and can be regenerated.
Complete the core and analysis stages before plotting, and generate all tables
before verification. Paths are resolved relative to the script, so calling
the script by its absolute path from another directory is supported.

## Configuration and Methods

Reference parameters are in `REF`; the core grid is in `core_grid()`;
phase-diagram and robustness scans are in `analysis()`.
Solver defaults specify iteration limits, tolerances, damping, and optimization
grids. Defaults preserve the paper's numerical settings. Changing them is a
new experiment, not reproduction of the supplied reference results.
The seed is fixed at 2024; the current computation uses no random sampling.

Equilibrium demand is solved by damped fixed-point iteration. Baseline pricing
uses grid search and golden-section search. Dual-track pricing uses a coarse
grid and multi-start coordinate golden-section search.
See [method mapping](docs/method_mapping.md) for equation correspondence,
the equation (13) notation discrepancy, and the table-label caveat.

## Repository Structure

```text
smart-healthcare-profit-model/
|-- reproduce.py
|-- reproduce_single.py
|-- README.md
|-- requirements.txt
|-- environment.yml
|-- LICENSE
|-- THIRD_PARTY_NOTICES.md
|-- .gitignore
|-- .gitattributes
|-- docs/
|   |-- method_mapping.md
|   `-- FINAL_CHECK.md
|-- tests/
|   `-- test_reproduction.py
`-- expected_outputs/
    |-- CHECKSUMS.txt
    `-- reference CSV and JSON files
```

## Results and Verification

The core experiment contains 10,368 valid combinations: 4,062 yield a profit
increase (39.18%) and 6,306 yield a profit decrease (60.82%).
The `verify` command checks byte-level MD5 agreement for all 14 reference
CSV/JSON files. MD5 is used for regression comparison, not security.
`.gitattributes` prevents Git from converting reference-data line endings.

Floating-point results may vary across platforms and numerical libraries.
Use the reference environment for exact matching. All six figures use English
labels and Times New Roman, with custom mathematical font settings.
Image pixels are not part of the numerical checksum comparison.

## License and GitHub Upload

MIT license; copyright 2026 张戈、卞慧馨、李佳路.
See `THIRD_PARTY_NOTICES.md` for dependency sources and licenses.
Include `expected_outputs/`; exclude `output/`, `__pycache__/`, and `.venv/`.
For browser uploads, exclude these directories manually as well.

```powershell
git init
git add .
git commit -m "Release reproducible numerical experiments"
git branch -M main
git remote add origin https://github.com/BHXHHHH/smart-healthcare-profit-model.git
git push -u origin main
```

These commands assume an uninitialized local repository and an empty GitHub
repository. Do not force-push over existing remote commits.
