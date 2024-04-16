# BOIS-with-immediate-sharing
This code was created as part of a bachelor's thesis in the field of mathematics. It extends the [BOIS for VQE](https://doi.org/10.1038/s41534-021-00452-9) algorithm by introducing "immediate sharing" and enabeling sharing between geometries whose hamiltonian are extended in differing sets of Pauli strings.

This repository includes the final code of the project as well as the raw data of the numerical experiments the corresponding paper (link added once published) is based on.

## Code description

The main algorithm can be found in the function "iBOIS" in the file "BOIS_based_VQE.py". 
Based on a specification how many gridpoint-layers per dimension, that can be changed in a hardcoded manner in the variable "grid_specification", and the molecules name, given when excecuting the program, the function "grid_to_molecules" crates a list of "MoleculeInfo" object representing the different geometries.
The objective function evaluation is handeled by the "ObjectiveFunction" class. 
Per geometry an ObjectiveFunction-object is created. During its initialisation, a Hartree-Fock calculation and the transfromations to finaly access the hamiltonian is its Pauli-expansion form are excecuted.
If the Pauli-expanision sets differ, this is handeled by the "twopaulis" function in the "pauli_handling.py" file. 
Based on these objects a ModularBayesianOptimization-object from the GPyOpt package is initialised.
The "parallel_BO_H2" excecutes the optimisation iterations until the convergence criterion is met and writes the output.

## Installation

   - Get yourself an [IBM Quantum account](https://quantum.ibm.com/) (There exists a free plan but with it you can access only part of this scrips functionality).
   - Place the Python files in a directory with a subdirectory "results".
   - Run 
        ```
        conda create --name ENV_iBOIS --file package-list.txt
        ```
     to replicate the environment the code was written in (or install the necessary packages manually). 
   - Insert your IBM token with access to Ehningen (if available) in line ???.
   - Insert your IBM token with access to the other IBM devices (if available) in line ??? and ???.

   Test the installation by running: 
   - "python BOIS_based_VQE.py 1 immediate_aa Matern52 no H2" for local calculation.
   - "nohup python BOIS_based_VQE.py 1 immediate_aa Matern52 FakeLima H2 >> test_connection.out &" for testing the connection to IBM runntime (due to queing this might take some time).

   (Note: Before changing "load_initp" to TRUE, make sure that reference folder (line ??? and ???) contains an output file with the corresponding repetition number.)

## Usage

Some of the calculations parameters are hardcoded and therefore need to be changed directly in the code if required. Those are:

- grid_specification (list): Includes the number of equally spaced points in each PES dimension of the grid investigated. Use e.g. [8] for H2 and [5,5] for H2O.
iterations (int): Maximum iteration number. Use e.g. 100 to reproduce the paper results.
- k_0 (int): Initial aquisition weight. In the paper k_0 = 1 was used.
- num_initp (int): Number of points to initialise the surrogate model. In the paper 30 was used.
- eps (float): Relatice deviation from the reference below which the optimisation stops.

Some specifications are passed to the code in form of command line arguments in the following order:

1. repetition_nr: Only relevant if you want to reuse a chertain set of input points. In that case using "load_initp = True" the input point with the same repetition number is loaded from the reference folder. If this is not relevant, any number can be chosen.
2. sharing_mode: Chose one of the sharing modes:

    - all-to-all-sharing: sharing partners include all geometries
    - no: no sharing takes place 
    - immediate_aa: newly calculated information is shared with all other geometries immediatly
    - NN: newly calculated information is shared with the nearest neighbours after each iteration
    - immediate_NN: newly calculated information is shared with the nearest neighbours immediatly
3. kernel_name: Chose either "RBF" or "Matern52".
4. backend_str: 
    - "no" for local calculation
    - "FakeLima" or "FakeKolkata" for fake devices
    - "Brisbane", "Osaka", "Kyoto" or "Ehningen" for real devices

5. molecule_name: "H2" or "H2O"

## License

## Data

## Contact

- Milena Röhrs ([Linkedin](https://de.linkedin.com/in/milena-roehrs))
- Supervisor: Arcesio Castaneda Medina ([Fraunhofer ITMW](https://www.itwm.fraunhofer.de/de/abteilungen/hpc/mitarbeiter/arcesio-castaneda-medina.html))

## References and Acknoldegements

The code was inspired by the [original BOIS code](https://github.com/chris-n-self/qc_optim/blob/master/qcoptim/bo_chemistry.py). This project was only possible due to the computation recources contributed by the Fraunhofer ITWM and IBM Quantum.
