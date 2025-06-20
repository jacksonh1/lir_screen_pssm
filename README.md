# lir_proteome_screen_pssm

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

developing a LIR-motif PSSM using data from a proteomic screen and evaluating its performance.



## installation

run `make` to see available commands.

to install the project, run:
- `make create_environment` to create a conda environment with the required dependencies.
    - make sure to then activate the environment with `conda activate lir_proteome_screen_pssm`
- `make data` to run the data processing scripts




## Project Organization

```
├── LICENSE                <- Open-source license if one is chosen
├── Makefile               <- Makefile with convenience commands like `make create_environment` or `make requirements`
├── README.md              <- The top-level README for developers using this project.
├── data
│   ├── processed          <- The final, canonical data sets for modeling.
│   └── raw                <- The original, immutable data dump.
│
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the date in yyyy-mm-dd format, and a short `_` delimited description, e.g.
│                         `01-2025-05-13-data_exploration.ipynb`. "DE" equals data exploration.
│
├── pyproject.toml         <- Project configuration file with package metadata for 
│                         lir_proteome_screen_pssm and configuration for tools like black
│
├── ref_materials          <- powerpoints + old code and such to draw from
│
├── reports                <- Generated analysis as HTML, PDF, LaTeX, ppt, etc.
│   └── figures            <- Generated graphics and figures to be used in reporting
│
├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g.
│                           generated with `pip freeze > requirements.txt`
│                         
├── processing_scripts     <- scripts for processing data. This is where the tools in `lir_proteome_screen_pssm` are actually applied.
│
└── lir_proteome_screen_pssm   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes lir_proteome_screen_pssm a Python module
    │
    ├── environment.py          <- Store useful variables and configuration
    │
    │── data_loaders.py         <- tools for loading the specific datasets into python
    │
    │── pssms.py                <- tools for working with PSSMs
    │
    │── sequence_utils.py       <- tools for working with sequences
    │
    └── ...                     <- ...
```

--------

