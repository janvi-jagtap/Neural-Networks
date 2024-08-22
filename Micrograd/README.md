# Micrograd
## A step-by-step explanation of backpropagation and training of neural networks using high school calculus and Python.

### Micrograd is a tiny deep learning library
- It is an autograd engine 
  - Autograd is short for automatic gradient
- It implements backpropagation 
  - Backpropagation is an algorithm that allows you to efficiently evaluate the gradient of some kind of a loss function with respect to the weights of a neural network
	    - Valuable because it allows you to tune the weights of the neural network to minimize the loss function which increases the accuracy of the network
  - Is the mathematical core of any deep neural network library (pytorch, jax)

*EXAMPLE FOR FUNCTIONALITY* 
  - A function where two values are inputted and then they're transformed into a new value by a bunch of functions
	- It will build out this entire math operation - expression graph
	- Keeps track of how everything is changed putting everything in variables and can go backwards through that expression graph and recursively applies chain rule from calculus - which means that you're doing the derivative of g with respect to the values
	- Neural networks are just mathematical expressions where the outputs are the prediction and are trained by backpropagation
	- These micrograds wouldn't work in production, in production you use tensors where none of the math changes, arrays of the scalar values - can then run it in parallel to make it more efficient

**Micrograd is all you need to train neural networks and a small level, everything else is for efficiency**


Definition of a derivative: 
- In simple words if you are at x and you increase or decrease by some h the value by how much y or the ouput changes is the derivative and the slope of the function at that point

### Manual back propagation through a simple expression and summary of Value Class
- These neural networks are large mathematical expressions
- Create Value class as a data structures to hold these mathematical expressions
- Use a tuple to keep track of the children of the mathematical expression as a field in the value class
- Use a string to keep track of the operation that the children did to produce the resulting value
- Use grad field to keep track of the derivative of the output with respect to its specific node value its assigned to, default is 0 (represents change which is why initially its 0 because we assume no change is made)
- In terms of the neural network some leaf nodes is a weight of the neural net and the derivative of the output in respect to the weights is the loss function which we want to minimize in order for accuracy
  - **"Taking the derivative of the output with respect to the different weights of the neural network." Means that in a big mathematical expression adding some amount of h to one weight of the neural network, the slope of the difference of the output divided by h is that derivative**
  - Other leaf nodes are data but don't need the derivative of that because data is fixed but the weights are iterated on using the gradient information
- Back propagation is the recursive multiplication done to get the output's derivative with respect to the weight that you're on. Simulating the chain rule
  - Its why the gradient is important because it influences the final output and is useful when training neural networks
	
### Neuron mathematical diagram
- Some x as an input
- Synapses that have weights
- The input that flows to the cell body is x * w
- In the cell body it takes the summation and then adds some bias
- Goes through some activation function that helps squash the inputs like tanh which caps off the inputs
- The output is its dot product

### Optimizing the manual back backpropagation
- First optimizing it for a single node
  - Create a backward function where the default is to do nothing
  - And then for each function adding, multiplying, and the tanh function it assigns the gradient to it's respective self and other nodes based on the local variable and children's grad using chain rule
- Now optimizing so that calling backwards on one node will recursively call it for all nodes part of the neuron <br />
  *Important thing to remember is that in order to back propagate on a node everything after it has to be done*
  - Use a topological sort which is a sort where all the nodes are pointing to the same way to the right
  - Then back propagate in reverse because remember we build from the back



### Relating what's been built to the PyTorch library
Available through the tensor objects in PyTorch
- Have to set required gradients because it's set automatically to no since leaf nodes with data won't have gradients
- Tensor objects have data values and grad values
- Have to call tensorname.data.item() to get just the value of the data
- Has a backward function to set all the gradients just like the one that we created
- It’s the exact same but much more efficient and can do more operations in parallel because of the tensor objects but conceptually the same

### Building out a neural net library in microgrid
- Neuron class which returns a single neuron (like the value class above)
    - Initialization method will have self and number of inputs
        - The weight and bias of self is randomly chosen
    - Call method is going to help so that we can create a neuron of like 2 dimension and then feed it x which is the 2 dimensional array
        - Want to multiply each weight and x value and add it together so use zip method in python which will pair up the tuples each w and x together
        - Then use a for loop to multiply each pair and then sum all of those up together
        - Then add self.b on top of it which is the bias
    - Then return the tanh of that value 

- Layer class: Neural networks aren't just one neuron, it's a layer of neurons which in other words is a list of neurons where the number of neurons is passed in as a parameter that aren't connected to each other but exist independently together and each one is connected to all the inputs
    - nin is the number of inputs
    - And nouts is the number of ouptuts like how many neurons you want in that layer
    - So like Layer(2, 3) is a layer of 3 neurons in a list where each neuron is created by 2 input neurons

- Multi-Layer Perceptron (MLP class)
    - nin is the number of inputs
    - Take multiple nouts which is the size of each layer that we want, essentiall a list of layer sizes (representing each layer)

### Using tiny dataset to understand the loss function
- Loss function is a single value that tells you how well the neural network is performing (we want to minimize it)
- Loss is calculated using outputs and gradients and so now when we look at each neuron's weight's gradients we can use that value to affect the loss function because can back propagate the loss function 
- Then update the data in the neurons based on the grad and iteratively update it and calculate the loss function making it go smaller each time 
