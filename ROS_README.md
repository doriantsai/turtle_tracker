# install mamba
conda install mamba -c conda-forge

# enter turtle env
mamba activate turtles

# this adds the conda-forge channel to the new created environment configuration 
conda config --env --add channels conda-forge
# and the robostack channel
conda config --env --add channels robostack-staging
# remove the defaults channel just in case, this might return an error if it is not in the list which is ok
conda config --env --remove channels defaults

# install ROS 
mamba install ros-noetic-desktop

# reactivate env
mamba deactivate
mamba activate turtles

# install ROS tools
mamba install compilers cmake pkg-config make ninja catkin_tools


