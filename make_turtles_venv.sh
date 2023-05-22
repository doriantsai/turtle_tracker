# script to automatically make conda environment defined in turtles.yml
mamba env create -f turtles.yml

conda activate turtles

pip install ultralytics
# often will need to pip isntall ultralytics -U # due to upgrades