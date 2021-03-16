
## 1. The class `neuralNetwork`

### 1.1. `__init__`(hyperparams (*dict.*))
Initializes parameters (weights and biases) and hyperparameters (learning rate, optimizer, etc.) of neural network

### 1.2. `setHyperparameters`(hyperparams (*dict.*))
Sets hyperparameters of neural network from provided input collection<br>
In order to standardize setting of parameters, a set of global constants have been declared just before the class declaration, which are used for matching the input hyperparameters with the ones for which functionality has been provided. Hence, it is advisabe to use their form while setting the input hyperparameters

### 1.3. `initModel`(hyperparams (*dict.*))
Initializes parameters (weight and bias matrices) of neural network

### 1.4. `activation`(layerNum, x)
Computes and return activation values for a given layer and its sum(a<sub>i</sub>) values<br>
**Note**: For activations requiring an exponential operation, it was observed - especially with hidden ReLU and output softmax layers - that when the magnitudes of each layer output increased, the exponential blew up very quickly. To prevent this, the functions were tackled as -
- For sigmoid, the input was clamped between two fixed high magnitudes beyond which the output was indistinguishable from 1 (order of magnitude of `x` ~ 100)
- for softmax, the value was shifted down by the maximum argument within the input and truncated beyond a certain tolerance value to that value itself

### 1.5. `activationDerivative`(layerNum,**kwargs)
Computes and returns activation derivative values for a given layer and its sum (a<sub>i</sub>) or output (h<sub>i</sub>) values depending on the given argument

### 1.6. `loss`(outputData,targetData)
Computes and returns loss values for given output and target data<br>
**Note**: For the cross-entropy derivative, due to the `log` operation, log-of-zero errors were encountered at some points. To fix this, the inputs are clamped between two values `EXP_INPUT_LOWER_TOL` and`EXP_INPUT_UPPER_TOL`

### 1.7. `lossOutputDerivative`(outputData,targetData)
Computes and returns loss derivatives for given output and target data<br>
**Note**: For the cross-entropy derivative, due to the division operation, divide-by-zero errors were encountered at many points. To fix this, the output values (in denominator) are clamped above the value `EXP_INPUT_LOWER_TOL` and`EXP_INPUT_UPPER_TOL`

### 1.8. `lossMetrics`(outputData,targetData)
Computes and returns loss metrics for given data as `(loss, accuracy)`

### 1.9. `forwardPass`(inputData)
Computes output activations of all layers of neural network<br>
Data can also be given as sets of datapoints (dimensions being layer dimension x dataset size - i.e. multiple columns with each column being a datapoint)

### 1.10. `backwardPass`(layerwiseOutputData, targetData)
Computes weight and bias gradients for all layers of neural network<br>
Data can also be given as sets of datapoints (dimensions being layer dimension x dataset size - i.e. multiple columns with each column being a datapoint)

### 1.11. `initOptimizerCollector`()
Creates object to store update variables used in memory-based algorithms (ex. `update` variables in mumentum)

### 1.12. `updateParameters`(inputData,targetData,opt):
Performs parameter updates for given input and target datapoints.<br>
`opt` is the optimizer object created by the above function.
- The reason for doing it in this roundabout fashion is to keep the training loop separate from the parameter update logic. This makes it possible for this function to be used in a different context too by an external user.

### 1.13. `infer`(inputData,**kwargs)
Perform inference on input dataset using the neural network<br><br>

(**Note** that unless `colwiseData=True` is given as an argument, data will be interpreted as being shape = (dataset size, layer dimension))<br>

Providing `targetData` as a kwarg also returns the output of `lossMetrics` as `(loss, accuracy)`, leading to the return value being `outputData,(loss, accuracy)`

### 1.14. `train`(inputData, targetData, x_val, y_val, **kwargs)
Train the network on the given input and target datasets.<br><br>

(**Note** that unless `colwiseData=True` is given as an argument, data will be interpreted as being shape = (dataset size, layer dimension))<br>

- Runs a training loop to perform minibatch gradient descent for one of the optimization algorithms according to the corresponding hyperparameter definition

Each optimizer also logs the losses and errors in each epoch with calls to `lossMetrics`.<br><br>

Note that `inputValidationData` and `targetValidationData` are given as an input just to calculate the loss and error at each epoch. They are ***strictly*** not used anywhere to train the neural network

### Notes on `neuralNetwork` class
- Gradient descent optimizations covered (can be found in `updateParameters` function):
1. Vanilla (unoptimized) gradient descent
```
# common processing
layerwiseOutputData = self.forwardPass(inputData)
(gradW, gradB)      = self.backwardPass(layerwiseOutputData,targetData)

# post-common processing
for i in range(1,self.numLayers+1):
	self.wmat[i] += -eta * gradW[i]
	self.bias[i] += -eta * gradB[i]
```

2. Momentun-based gradient descent
```
# common processing
layerwiseOutputData = self.forwardPass(inputData)
(gradW, gradB)      = self.backwardPass(layerwiseOutputData,targetData)

# post-common processing
for i in range(1,self.numLayers+1):
	update_w[i] = gamma*update_w[i] + eta*gradW[i]
	update_b[i] = gamma*update_b[i] + eta*gradB[i]
	self.wmat[i] += -update_w[i]
	self.bias[i] += -update_b[i]
```

3. Nesterov-accelerated gradient descent
```
# pre-common processing
for i in range(1,self.numLayers+1):
	self.wmat[i] += -gamma*update_w[i]
	self.bias[i] += -gamma*update_b[i]

# common processing
layerwiseOutputData = self.forwardPass(inputData)
(gradW, gradB)      = self.backwardPass(layerwiseOutputData,targetData)

# post-common processing
for i in range(1,self.numLayers+1):
	update_w[i] = gamma*update_w[i] + eta*gradW[i]
	update_b[i] = gamma*update_b[i] + eta*gradB[i]
	self.wmat[i] += -eta*gradW[i]
	self.bias[i] += -eta*gradB[i]
```

4. RMSprop
```
# common processing
layerwiseOutputData = self.forwardPass(inputData)
(gradW, gradB)      = self.backwardPass(layerwiseOutputData,targetData)

# post-common processing
for i in range(1,self.numLayers+1):
	v_w = opt["v_w"]; v_b = opt["v_b"]
	v_w[i] = beta*v_w[i] + (1-beta)*gradW[i]**2
	v_b[i] = beta*v_b[i] + (1-beta)*gradB[i]**2 
	self.wmat[i] += -eta * (v_w[i] + epsilon)**-0.5 * gradW[i]
	self.bias[i] += -eta * (v_b[i] + epsilon)**-0.5 * gradB[i]
```

5. AdaM
```
# common processing
layerwiseOutputData = self.forwardPass(inputData)
(gradW, gradB)      = self.backwardPass(layerwiseOutputData,targetData)

# post-common processing
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

6. NAdaM
```
# common processing
layerwiseOutputData = self.forwardPass(inputData)
(gradW, gradB)      = self.backwardPass(layerwiseOutputData,targetData)

# post-common processing
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

- To add a new optimizer:
1. Add the core math functionality in an `elif` statement in the `updateParameters` function, similar to the above mentioned algorithms
2. If any memory storage is required, it can be declared in the `initOptimizerCollector` function
3. Put the algorithm `if`-clause key above the class, naming it similar to the already named ones. You're done!

- To add a new loss function:
1. Add the loss and loss derivative functionality in `loss` and `lossOutputDerivative` functions respectively
2. Put the loss function `if`-clause key above the class, naming it similar to the already named ones. You're done!

- To add a new activation function:
1. Add the activation function and activation derivative functionality in `activation` and `activationDerivative` functions respectively
2. Put the activation function `if`-clause key above the class, naming it similar to the already named ones. You're done!


## 2. Reading & processing the data
`keras.datasets.fashion_mnist.load_data()` returns `( (x_train, y_train), (x_test, y_test) )`
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
- `regparam`: L2 regularisation (weight decay) coefficient (*alpha*)
- `wandb`: whether WandB is to be used or not (Boolean)

### 3.2. Training and evaluation
A `neuralNetwork` object is created with the constructor argument being the above created dict obect that holds the hyperparameters. Once that is done, the `train` member function of the class is invoked with the arguments being the training inputs and targets, as well as the validation inputs and targets, which are **strictly** not used for any training, only in the calculation of loss. At each epoch, the `lossMetrics` function logs training and validation loss and accuracy values into a WandB run.

Once the model has been trained, test data is run using the `infer` function. Error can be obtained by passing the target data as a `targetData` kwarg and receiving the outputs as: `outputData, (loss, accuracy)` (without passing the kwarg, only `outputData` is returned).

## 4. Executing a sweep
Finally, in order to run the sweep, a function `runSweep()` is provided (not part of class). This function translates the parameters as given in the assignment report into the forms compatible with our model (keeping in mind the need for a higher level of generality), performs all the training runs and logs all the values mentioned in the earlier section into WandB.

## 5. Testing and Confusion Matrix
In order to plot the confusion matrix, the data is inferred for a set of hyperparameters that displayed a high accuracy during the sweeps. Once that is done, `matplotlib qt` and `mpl_toolkits` are used to plot the confusion matrix. We hope you like our humble effort of trying to be a little creative with it.

## 6. MNIST digit database runs
Finally, three sets of hyperparameters showing the most optimal loss metrics are applied to the problem of training the MNIST digit database. The procedure is fairly similar to that employed in the Fashion-MNIST dataset used upto now, and the results are logged into WandB.
