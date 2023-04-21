# turtle_tracker
To automatically count sea turtles and birds from drone imagery, we have developed a turtle and bird detector, classifier and tracker.

## Installation & Dependencies
- The following code was run in Ubuntu 20.04 LTS using Python 3
- Install conda https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html, although I recommend miniforge/mambaforge https://github.com/conda-forge/miniforge, which is a much more minimal and faster installer for conda packages. These installers will create a ''base'' environment that contains the package managers for conda.
- `conda install mamba -c conda-forge`
- `git clone git@github.com:doriantsai/turtle_tracker.git`
- `cd` to the turtle_tracker folder
- Create conda environment automatically by running the script "make_conda_turtles.sh" file
    - make .sh file executable via `chmod +x make_conda_turtles.sh`
    - run .sh file via `./make_conda_turtles.sh`
    - Note: you can read the dependencies in the `turtles.yml` file
- this should automatically make a virtual environment called "agkelpie"
- to activate new environment: `conda activate turtles`
- to deactivate the new environment: `conda deactivate`

## WandB
- Weights & Biases is used to track and visualise the model training progress
- Setup an account https://docs.wandb.ai/quickstart