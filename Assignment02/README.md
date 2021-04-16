# Assignment 2 - Convolutional Neural Networks

# A. Training, evaluating and testing models with hyperparameter sweeps
## Functions
1. `model = initModel(config)`
Creates and returns Keras model for given hyperparameter configuration

2. `trainingRun(config, **kwargs)`
Performs model fitting for given network configuration and data (given as kwargs).
Data can be given in 2 ways:
* `trainingRun(runCfg, xdata=xdata, ydata=ydata)`
Data is given as inputs and targets, with validation split obtained from the training params within the run config passed
* `trainingRun(runCfg, train_data=train_data, val_data=val_data)`
Data is given as training and validation data directly, the split being done externally by the user themselves

3. `train_dataset, val_dataset = I12kDatasets(config)`
Processes and returns the iNaturalist-12K dataset to be used for training and validating the CNNs. Data is returned in the form of training and validation datasets as shown above. Config is provided to initialize dataset functions for the parameters to be used such as batch size and validation split. 

4. `image, labels = augmentImage(image, labels)`
Tensorflow data pipeline function used for data augmentation. Performs random flipping of images vertically and horizontally

5. `config = configFromJSON(arg)`
Loads and returns runtime config from provided JSON file. Arg given is JSON file name, file descriptor of the JSON file or the processed configuration itself

6. `config = configForA2WandbSweep()`
Returns config to be used for run that is part of a WandB sweep, specific to the architecture to be used for the current assignment

7. `runWandbSweep()`
Harness for performing WandB sweep that uses above function(s) to perform a sweep run

8. `test_ds = I12kDatasets_test(config)`
Processes and returns the iNaturalist-12K dataset to be used for testing the CNNs.

## More about the config passed to every function
The config is essentially a dictionary object that contains information about the network architecture and training parameters suchb as number of epochs, batch size and so on. Here are the keys that it employs:
* `layers`: list of layer configs in the following format:
    * `type`: String, layer type, such as `conv`,`fc`,`flatten`,`dropout` and so on
    * parameters specific to the layer type given above
* `trainparams`: parameters used in training, such as:
    * `epochs`: Int
    * `batch_size`: Int
    * `val_split`: Float
    * `dsAugment`: Boolean
* `wandb`: Boolean, whether WandB is used or not

# B. Finetuning weights for an existing architecture
## Functions
1. `finetune_model(hyp)`
Performs finetuning on the given model for the given set of the following training parameters:
* `model`: neural network architecture to be used
* `include_top`: whether to include the 1000-classes layer between CONV and final 10-class layer or not
* `num_unfrozen`: number of CONV layers before dense layer to be unfrozen for the second half of training
* `eta`: learning rate
* `dropout`: pre-dense layer dropout fraction
* `epochs`: number of epochs

2. `runSweep()`
Harness for performing WandB sweep that uses above function(s) to perform a sweep run

# C. YOLO-based object detection
[Link to dataset](https://www.kaggle.com/aditya276/face-mask-dataset-yolo-format)<br>
[Link to darknet repository](https://github.com/AlexeyAB/darknet)<br>
[Link to sample video](https://youtu.be/pnCPKJ0zHeo)
