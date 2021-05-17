# FCNN
FCNN is a Full Connected Neural Network.
NN.py contains the class Network and NN1.py is an example of using this class.
In the constructor of the class you can specify the number of nodes in input,hidden and output layer.
Example1: net = nn.Network([2, 3, 1]) is an FCNN of 2 input nodes, hidden layer1 has 3 nodes and output layer 1 node.
Example2: net = nn.Network([100, 300, 200, 10]) is an FCNN of 100 input nodes, hidden layer1 has 300 nodes, hidden layer2 has 200 nodes and output layer 10 nodes.
This FCNN has a non-linear Sigmoid activation function.
You can easy use this class Network to configure your own neural network.

To use the Example NN1.py
    git clone https://github.com/redingen/FCNN.git
    cd FCNN
    source virtual/bin/activate
    python3 NN1.py
    close each time displayed figure to continue
    deactivate
    
