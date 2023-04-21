# script to automatically make conda environment defined in turtles.yml
mamba env create -f turtles.yml

conda activate turtles

pip install thop
