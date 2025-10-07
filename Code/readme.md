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
```
├── Data/ # Output of the reconstructions
│ ├── LA/
│ ├── LAP_Phantom/
│ ├── PAT/
│ └── ROI/

├── problems/ # Modules for reconstruction and data generation
│ ├── problem_ct.py # Classic Radon transform
│ └── problem_lap.py # Circular mean Radon transform

├── source/ # Main scripts
│ ├── make_plot.py # Generate SVG plots used in the thesis
│ ├── phantom_ct.py # Example: Reconstruction of Head Phantom (Radon transform)
│ ├── run_ct.py # Run Radon transform reconstruction for rectangles/ellipses
│ └── run_pat.py # Run circular mean Radon transform reconstruction
```
