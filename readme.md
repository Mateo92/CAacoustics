

# Description

This code is for propagating the acoustic waves from a vibrating solid in open space, by discreticing the solid surfece into spherical sources and propagating each source contribution with a cellular automaton. 

It conatains a code to simulate a beam vibrating; using the Euler-Bernulli model and solved it via normal modes. As well as a FEniCS script to simulate the vibration of a xilophone/marimba-like elastoplastic solid. These codes generate the objects from witch the cellular automaton will propagate its sound.



# Installation

The recommended way to install this code is to install python and the modules specified later.



# Usage

To run the simulations we recommend to clone this repository entirely, then try to run the Notebooks examples in the directory they are in.

# Dependencies

python packages, from pip
- pyyaml
- pandas
- imageio
- progressbar
- numpy

FEniCS, to generate data of a vibrating solid.

Recommended installation commands
```bash
pip install --user pyyaml pandas imageio progressbar
```

# Caveats


This code generates a lot of csv files and .gif files. Todo so, it creates folders and store them.

Please install dependencies.

# Acknowledgements

- comet-fenics: https://comet-fenics.readthedocs.io/en/latest/intro.html
- FEniCS

