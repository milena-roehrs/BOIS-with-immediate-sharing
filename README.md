# BOIS-with-immediate-sharing
This code was created as part of a bachelor's thesis in the field of mathematics. It extends the [BOIS for VQE](https://doi.org/10.1038/s41534-021-00452-9) algorithm by introducing "immediate sharing" and enabeling sharing between geometries whose hamiltonian are extended in differing sets of pauli strings.

This repository includes the final code of the project as well as the raw data of the numerical experiments the corresponding paper (link added once published) is based on.

## Code description

The main algorithm can be found in the function "parallel_BO_H2" in the file "BOIS_based_VQE.py". 
Based on a specification how many gridpoint-layers per dimension, that can be changed in a hardcoded manner in the variable "grid_specification", and the molecules name, given when excecuting the program, the function "grid_to_molecules" crates a list of "MoleculeInfo" object representing the different geometries.
The objective function evaluation is handeled by the "ObjectiveFunction" class. 
Per geometry an ObjectiveFunction-object is created. During its initialisation, a Hartree-Fock calculation and the transfromations to finaly access the hamiltonian is its Pauli-expansion form are excecuted.
If the Pauli-expanision sets differ, this is handeled by the "twopaulis" function in the "pauli_handling.py" file. 
Based on these objects a ModularBayesianOptimization-object from the GPyOpt package is initialised.
The "parallel_BO_H2" excecutes the optimisation iterations until the convergence criterion is met and writes the output.

## Installation

   - Get yourself an IBM Quantum account at https://quantum.ibm.com/ (There exists a free plan put then you can access only part of this scrips functionality).
   - Place the Python files in a directory with a subdirectory "results".
   - Run 
        ```
        conda create --name ENV_iBOIS --file package-list.txt
        ```
     to replicate the environment the code was written in (or install the necessary packages manually). 
   - Insert your IBM token with access to Ehningen (if available) in line ???.
   - Insert your IBM token with access to the other IBM devices (if available) in line ??? and ???.

   Test the installation by running: 
   - "???" for local calculation
   - "???" for testing the connection to IBM runntime

   (Note: Before changing "load_initp" to TRUE, make sure that reference folder (line 386 and 482) contains an output file with the corresponding repetition number.)

## Usage

## License

## Data

## Contributions

## Contact

## References