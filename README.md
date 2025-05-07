utility.py: some useful tools for other functions
deim.py: find the sensor locations inductively
main.py: the main driver to train the neural network (based on the pytorch-lightning library)
grid_search.sh: perform a grid search for different hyper-parameters of the neural network
deim_nn_model.py: setup the neural network architecture
deim_nn_data.py: prepare the data for training, validation, and testing
deim_nn_aero_coeff.py: read the checkpoint which contains the parameters of the neural network after training and perform prediction on testing dataset
epoch=18-step=2584-val_loss=1.287397e-01.ckpt: the checkpoint corresponding to 5 sensors for 2D NACA0015 airfoil

