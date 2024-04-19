# BOIS-with-immediate-sharing
This code was created as part of a bachelor's thesis in the field of mathematics. It extends the [BOIS for VQE](https://doi.org/10.1038/s41534-021-00452-9) algorithm by introducing "immediate sharing" and enabling sharing between geometries whose Hamiltonian are extended in differing sets of Pauli strings.

This repository includes the final code of the project as well as the raw data of the numerical experiments the corresponding paper (link added once published) is based on.

## Code description

The main algorithm can be found in the function "iBOIS" in the file "BOIS_based_VQE.py". Based on a specification how many gridpoint-layers per dimension, that can be changed in a hardcoded manner in the variable "grid_specification", and the molecules name, given when executing the program, the function "grid_to_molecules" crates a list of "MoleculeInfo" object representing the different geometries. The objective function evaluation is handled by the "ObjectiveFunction" class. Per geometry an ObjectiveFunction-object is created. During its initialisation, a Hartree-Fock calculation and the transformations to finally access the Hamiltonian is its Pauli-expansion form are executed. If the Pauli-expansion sets differ, this is handled by the "twopaulis" function in the "pauli_handling.py" file. Based on these objects a ModularBayesianOptimization-object from the GPyOpt package is initialised. The "parallel_BO_H2" executes the optimisation iterations until the convergence criterion is met and writes the output.

## Installation

   - Get yourself an [IBM Quantum account](https://quantum.ibm.com/) (There exists a free plan but with it you can access only part of this scrips functionality). Put the (free plan) token in line 1062.
   - Place the Python files in a directory with a subdirectory "results".
   - Run 
        ```
        conda create --name ENV_iBOIS --file package-list.txt
        ```
     to replicate the environment the code was written in (or install the necessary packages manually). 
   - Insert your IBM token with access to Ehningen (if available) in line 1112.
   - Insert your IBM token with access to the other IBM devices (if available) in line 1062 and 1122.

   Test the installation by running: 
   - "python BOIS_based_VQE.py 1 immediate_aa Matern52 no H2" for local calculation.
   - "nohup python BOIS_based_VQE.py 1 immediate_aa Matern52 FakeLima H2 >> test_connection.out &" for testing the connection to IBM runtime (due to queuing this might take some time).

   (Note: Before changing "load_initp" to TRUE, make sure that reference folder (line 631 and 724) contains an output file with the corresponding repetition number.)

## Usage

Some of the calculations parameters are hardcoded and therefore need to be changed directly in the code if required. Those are:

- grid_specification (list): Includes the number of equally spaced points in each PES dimension of the grid investigated. Use e.g. [8] for H2 and [5,5] for H2O.
iterations (int): Maximum iteration number. Use e.g. 100 to reproduce the paper results.
- k_0 (int): Initial acquisition weight. In the paper k_0 = 1 was used.
- num_initp (int): Number of points to initialise the surrogate model. In the paper 30 was used.
- eps (float): Relative deviation from the reference below which the optimisation stops.

Some specifications are passed to the code in form of command line arguments in the following order:

1. repetition_nr: Only relevant if you want to reuse a certain set of input points. In that case using "load_initp = True" the input point with the same repetition number is loaded from the reference folder. If this is not relevant, any number can be chosen.
2. sharing_mode: Chose one of the sharing modes:

    - all-to-all-sharing: sharing partners include all geometries
    - no: no sharing takes place 
    - immediate_aa: newly calculated information is shared with all other geometries immediately
    - NN: newly calculated information is shared with the nearest neighbours after each iteration
    - immediate_NN: newly calculated information is shared with the nearest neighbours immediately
3. kernel_name: Chose either "RBF" or "Matern52".
4. backend_str: 
    - "no" for local calculation
    - "FakeLima" or "FakeKolkata" for fake devices
    - "Brisbane", "Osaka", "Kyoto" or "Ehningen" for real devices

5. molecule_name: "H2" or "H2O"

## Remarks on the convergence criterion

Choosing a fixed number of iterations $n_{max}$ for the Bayesian optimisation is the default stopping criterion in standard implementations such as GPyOpt. More elaborate stopping criteria exist, but are not the focus of this work. Instead, the objective was to determine the quantum computing (QC) demand that is needed to get sufficiently close to the exact Variational Quantum Eigensolver (VQE) solution. For this purpose, the calculation was stopped when the relative deviation from the reference was smaller than a predefined value. 

For the $\text{H}_2$ molecule, $10^{-3}$ was chosen as this value, which is loosely oriented on the concept of "chemical accuracy" (as the absolute values of the energies are around 1 Hartree). In the subsequent $\text{H}_2\text{O}$ experiments, this accuracy could not be reached within a reasonable number of iterations, which is why $2\cdot10^{-3}$ was chosen as a convergence criterion, instead.

## License

## Data

The raw data output of the numerical experiments for the $H_2$ and $H_2O$ molecule can be found in the folders "H2_raw_data" and "H2O_raw_data" respectively which include subfolders which are each associated with one of the subsections in the "Results and Discussion" section of the paper.

## Contact

- Milena Röhrs ([Linkedin](https://de.linkedin.com/in/milena-roehrs))
- Supervisor: Arcesio Castaneda Medina ([Fraunhofer ITMW](https://www.itwm.fraunhofer.de/de/abteilungen/hpc/mitarbeiter/arcesio-castaneda-medina.html))

## References and Acknowledgements

The code was inspired by the [original BOIS code](https://github.com/chris-n-self/qc_optim/blob/master/qcoptim/bo_chemistry.py). This project was only possible due to the computation recources contributed by the Fraunhofer ITWM and IBM Quantum.
