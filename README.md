### CS6910, Spring 2021
# Assignment 01
### Krish R. Kansara (EP17B005), Srijan Gupta (EP17B009)

## 1. The class `neuralNetwork`

### 1.1. `__init__`(hyperparams (*dict.*))
Initializes parameters (weights and biases) and hyperparameters (learning rate, optimizer, etc.) of neural network

### 1.2. `setHyperparameters`(hyperparams (*dict.*))
Sets hyperparameters of neural network from provided input collection

### 1.3. `initModel`(hyperparams (*dict.*))
Initializes parameters (weight and bias matrices) of neural network

### 1.4. `activation`(layerNum, x)
Computes and return activation values for a given layer and its sum(a<sub>i</sub>) values

### 1.5. `activationDerivative`(layerNum,**kwargs)
Computes and returns activation derivative values for a given layer and its sum (a<sub>i</sub>) or output (h<sub>i</sub>) values depending on the given argument

### 1.6. `lossOutputDerivative`(outputData,targetData)
Computes and returns loss derivatives for given output and target data

### 1.7. `forwardPass`(inputData)
Computes output activations of all layers of neural network
Data can also be given as sets of datapoints (dimensions being layer dimension x dataset size - i.e. multiple columns with each column being a datapoint)

### 1.8. `backwardPass`(layerwiseOutputData, targetData)
Computes weight and bias gradients for all layers of neural network
Data can also be given as sets of datapoints (dimensions being layer dimension x dataset size - i.e. multiple columns with each column being a datapoint)

### 1.9. `infer`(inputData,**kwargs)
Perform inference on input dataset using the neural network
Note that unless `colwiseData=True` is given as an argument, data will be interpreted as being dataset size x layer dimension

** Functions 1.10. & 1.11. are called in the functions following them (1.12. to 1.17.)** 

### 1.10. `gradtheta_for_batchindex`(inputData, targetData, datasetSize, batchSize, numBatches, batchIndex)
- Creates the input and target batch for the current batchIndex
- Gets the output for all layers by doing `forwardPass`
- Calculates the gradients w.r.t. the weights and biases by doing `backwardPass` and returns them

### 1.11. `update_val_train_loss_and_acc`(inputData, targetData, x_val, y_val, epoch)
- Calculates the output for the complete input data
- Calculates the squared error/cross entropy loss according to the definition of the corresponding hyperparameter for both training and validation sets
- Calculates and adds the l2 regularization loss
- Calculates the zero-one error
Appends the losses and the errors to the corresponding lists
- Logs the losses and errors and epoch no. in WandB.

### 1.12. `sgd`(inputData, targetData, datasetSize, batchSize, numBatches, x_val, y_val)
- Performs ‘vanilla’ stochastic gradient descent. Update rule for each batch:
```
#Get grad theta
(gradW, gradB) = self.gradtheta_for_batchindex(inputData, targetData, datasetSize, batchSize, numBatches, batchIndex) ##
# perform parameter update
for i in range(1,self.numLayers+1):
  self.wmat[i] += -self.learningRate * gradW[i]
  self.bias[i] += -self.learningRate * gradB[i]
```
- Calls `update_val_train_loss_and_acc(...)` in each epoch.

### 1.13. `momentumGD`(inputData, targetData, datasetSize, batchSize, numBatches, x_val, y_val)
-  Performs momentum based gradient descent. Update rule for each batch:
```
#Get grad theta
(gradW, gradB) = self.gradtheta_for_batchindex(inputData, targetData, datasetSize, batchSize, numBatches, batchIndex) ##
# perform parameter update
for i in range(1,self.numLayers+1):
  update_w[i] = gamma*update_w[i] + eta*gradW[i]
  update_b[i] = gamma*update_b[i] + eta*gradB[i]
  self.wmat[i] += -update_w[i]
  self.bias[i] += -update_b[i]
```
- Calls `update_val_train_loss_and_acc(...)` in each epoch.

### 1.14. `NAG`(inputData, targetData, datasetSize, batchSize, numBatches, x_val, y_val)
- Performs nesterov accelerated gradient descent. Update rule for each batch:
```
# perform look ahead parameter update
for i in range(1,self.numLayers+1):
  self.wmat[i] += -gamma*update_w[i]
  self.bias[i] += -gamma*update_b[i]
(gradW, gradB) = self.gradtheta_for_batchindex(inputData, targetData, datasetSize, batchSize, numBatches, batchIndex) ##
# perform parameter update
for i in range(1,self.numLayers+1):
  update_w[i] = gamma*update_w[i] + eta*gradW[i]
  update_b[i] = gamma*update_b[i] + eta*gradB[i]
  self.wmat[i] += -eta*gradW[i]
  self.bias[i] += -eta*gradB[i]
```
- Calls `update_val_train_loss_and_acc(...)` in each epoch.

### 1.15. `rmsprop`(inputData, targetData, datasetSize, batchSize, numBatches, x_val, y_val)
- Performs root mean square propagation.  Update rule for each batch:
```
(gradW, gradB) = self.gradtheta_for_batchindex(inputData, targetData, datasetSize, batchSize, numBatches, batchIndex) ##
# perform parameter update
for i in range(1,self.numLayers+1):
  v_w[i] = beta*v_w[i] + (1-beta)*gradW[i]**2
  v_b[i] = beta*v_b[i] + (1-beta)*gradB[i]**2 
  self.wmat[i] += -eta * (v_w[i] + epsilon)**-0.5 * gradW[i]
  self.bias[i] += -eta * (v_b[i] + epsilon)**-0.5 * gradB[i]
```
- Calls `update_val_train_loss_and_acc(...)` in each epoch.

### 1.16. `adam`(inputData, targetData, datasetSize, batchSize, numBatches, x_val, y_val)
- Performs adam. Update rule for each batch:
```
(gradW, gradB) = self.gradtheta_for_batchindex(inputData, targetData, datasetSize, batchSize, numBatches, batchIndex) ##
# perform parameter update
for i in range(1,self.numLayers+1):
  m_w[i] = beta_1*m_w[i] + (1-beta_1)*gradW[i]
  m_b[i] = beta_1*m_b[i] + (1-beta_1)*gradB[i]
  v_w[i] = beta_2*v_w[i] + (1-beta_2)*gradW[i]**2
  v_b[i] = beta_2*v_b[i] + (1-beta_2)*gradB[i]**2
  m_w_hat = m_w[i]/(1-beta_1**t)
  m_b_hat = m_b[i]/(1-beta_1**t)
  v_w_hat = v_w[i]/(1-beta_2**t)
  v_b_hat = v_b[i]/(1-beta_2**t)
  self.wmat[i] += -eta * (v_w_hat + epsilon)**-0.5 * m_w_hat
  self.bias[i] += -eta * (v_b_hat + epsilon)**-0.5 * m_b_hat
```
- Calls `update_val_train_loss_and_acc(...)` in each epoch.

### 1.17. `nadam`(inputData, targetData, datasetSize, batchSize, numBatches, x_val, y_val)
- Performs nadam. Update rule for each batch:
```
(gradW, gradB) = self.gradtheta_for_batchindex(inputData, targetData, datasetSize, batchSize, numBatches, batchIndex) ##
# perform parameter update
for i in range(1,self.numLayers+1):
  m_w[i] = beta_1*m_w[i] + (1-beta_1)*gradW[i]
  m_b[i] = beta_1*m_b[i] + (1-beta_1)*gradB[i]
  v_w[i] = beta_2*v_w[i] + (1-beta_2)*gradW[i]**2
  v_b[i] = beta_2*v_b[i] + (1-beta_2)*gradB[i]**2
  m_w_hat = (beta_1/(1-beta_1**(t+1)))*m_w[i] + ((1-beta_1)/(1-beta_1**t))*gradW[i]
  m_b_hat = (beta_1/(1-beta_1**(t+1)))*m_b[i] + ((1-beta_1)/(1-beta_1**t))*gradB[i]
  v_w_hat = v_w[i]/(1-beta_2**t)
  v_b_hat = v_b[i]/(1-beta_2**t)
  self.wmat[i] += -eta * (v_w_hat + epsilon)**-0.5 * m_w_hat
  self.bias[i] += -eta * (v_b_hat + epsilon)**-0.5 * m_b_hat
```
- Calls `update_val_train_loss_and_acc(...)` in each epoch.

### 1.18. `train`(inputData, targetData, x_val, y_val, **kwargs)
Train the network on the given input and target datasets.

(**Note** that unless `colwiseData=True` is given as an argument, data will be interpreted as being shape = (dataset size, layer dimension))

- Performs one of the optimization algorithms (from 1.12. to 1.17.) according to the corresponding hyperparameter definition to 'improve' the weights and biases.

(Note that each optimizer also logs the losses and errors in each epoch).

## 2. Reading & processing the data
`keras.datasets.fashion_mnist.load_data()` returns ( (x_train, y_train), (x_test, y_test) )
### 2.1. Processing 'x' data:
The elements of x_train and x_test are 2D numpy arrays (corresponding to each image) but our neural network can only take a 1D input. Hence, we flatten each of the elements by using `np.reshape(...)`. The processed datasets are named `x_train_1D` and `x_test_1D`.
```
len_1D = x_train.shape[1]*x_train.shape[2]
x_train_1D = np.array( [x.reshape(len_1D) for x in x_train] )
```

### 2.2. Processing 'y' data:
Each of the elements in y data are scalar labels (0 to 9) corresponding to the 10 classes. For our neural network it is easier to work with y data where each element is a vector **e**_i (length = no. of classes, with 1 at i<sup>th</sup> place, 0 otherwise) where 'i' is the label of the element. The transformed data is named `y_train_1D` and `y_test_1D` (because now each element is a 1D array instead of a scalar)
```
for i in range(len(y_train)):
  y_train_1D[i, y_train[i]] = 1
```

### 2.3. Splitting train data into train2 and validation data
- Define the `frac_val`, the fraction of data to be used for validation.
- Choose `frac_val*len(x_train_1D)` indices using `np.random.choice(...)` without replacement among all the indices for `x_train_1D` to define the validation set. Define the train2 set using rest of the indices.

## 3. Training and evaluation on an instance of hyperparameter
### 3.1. Defining the hyperparameters
Hyperparameters are defined as a dictionary object, with the keys and values being the names of and the values taken by the hyperparameters. For instance, an example hyperparameter object would be:
```
hyp = {
  "epochs": 20,
  "learningRate": 1e-3,
  ...
}
```
Here are the currently supported hyperparameters for our neural network class:
- `layerSizes`: List of number of nodes per layer, with the first value being number of inputs and the last value being the number of outputs: Example for Fashion-MNIST: `[784,128,64,10]`
- `batchSize`: Size of batch for minibatch gradient descent
- `learningRate`: Learning rate of model
- `epochs`: Number of epochs for which to train model
- `activations`: List of activation functions for each layer. For the above example: `[ACTIVATION_TANH, ACTIVATION_SIGMOID, ACTIVATION_SOFTMAX]`
- `lossFn`: Loss function used by model. Options include `LOSS_SQERROR` and `LOSS_CROSSENTROPY`
- `initWeightMethod`: Method of initial weight initialization. Options include `WINIT_RANDOM` and `WINIT_XAVIER`
- `initWeightBounds`: Initial weight bounds for `WINIT_RANDOM`
- `optimizer`: Backpropagation optimization used. Options include `GDOPT_NONE`, `GDOPT_MOMENTUM`, `GDOPT_NESTEROV`, `GDOPT_RMSPROP`, `GDOPT_ADAM` and `GDOPT_NADAM`
- `beta_1`: momentum scaling parameter (*gamma* in momentum and Nesterov optimizations, *beta_1* in AdaM and NAdaM)
- `beta_2`: learning rate scaling hyperparam (used in AdaM and NAdaM)
- `epsilon`: learning rate scaling hyperparam (used in AdaM and NAdaM)
- `regparam`: L2 regularisation coefficient (*alpha*)

### 3.2. Training and evaluation
A `neuralNetwork` object is created with the constructor argument being the above creted dict obect that holds the hyperparameters. Once that is done, the `train` member function of the class is invoked with the arguments being the training inputs and targets, as well as the validation inputs and targets, which are **strictly** not used for any training, only in the calculation of loss. At each epoch, the `update_val_train_loss_and_acc` functions logs training and validation loss and accuracy values into a WandB run.

Once the model has been trained, test data is run using the `infer` function. Error logging can be done indirectly by passing the test data to the earlier function and observing the WandB run status.

## 4. Executing a sweep
Finally, in order to run the sweep, a function `runSweep()` is created (not part of class). This function translates the parameters as given in the assignment report into the forms compatible with our model (to maintain generality), performs all the training runs and logs all the values mentioned in the earlier section into WandB.
