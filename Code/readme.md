This is the code-implementation of my Master Thesis with the topic:
"Zur Charakterisierung von Artefakten in der Tomographie mit unvollständigen Daten"

# Instalation
## 1. git Repository 
```bash
    git clone https://github.com/TuschnerF/MasterThesisCode.git
```
## Instalation of requierd Packages
```bash
    conda env create -f environment.yaml
```

# Code overview
.
├── Data/                    # Output of the reconstructions

│   ├── LA/

│   ├── LAP_Phantom/

│   ├── PAT/

│   └── ROI/

├── problems/                # 

|   ├── problem_ct.py        # Functions for reconstruction and data generation for the classic Radon transform

|   ├── problem_lap.py       # Functions for reconstruction and data generation for the circular mean Radon transform

├── source/                  # Main scripts

│   ├── make_plot.py         # Script for generating the svg-plots used in the thesis

│   ├── phantom_ct.py        # Example: Reconstruction of a Head-Phantom for the Radon transform

│   ├── run_ct.py            # script for running reconstruction of Radon transform for rectangles and ellipses

│   ├── run_pat.py           # script fpr running reconstruction of circular mean Radon transform
