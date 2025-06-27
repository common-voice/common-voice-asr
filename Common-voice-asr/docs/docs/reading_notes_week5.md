# PyTorch Hyperparameter Tuning Tutorial
- hyperparameter tuning - choosing a different learning rate or changing a network layer size can have a dramatic impact on model performance
Ray Tune = tool for parameter tuning
 - integrate through slight mods:
    - wrap data loading and training in functions
    - make some network parameters configurable
    - add checkpointing (optional)
    - define the search space for the model tuning
- install package: ray[tune] & torchvision
    - from ray import tune
    - from ray import train
     - from ray.train import Checkpoint, get_checkpoint
    - from ray.tune.schedulers import ASHAScheduler
    - import ray.cloudpickle as pickle
Dataloaders
 - wrap in their own func & pass a global dir - can share data dir between diff trials
CNN
 - can only tune parameters that are configurable
Train function
- wrap in function train_cifar(config, data_dir=None)
    - config = hyperparameters to train w
    - data_dir = specifies where we load and store the data, so multiple runs can share the same data source
- learning rate also configurable: optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)
- split training data into validation & training -> train on 80% & calculate val loss on remaining 20%
- batch sizes configurable
Adding (multi) GPU support with DataParallel
- can wrap model in nn.DataParallel to support data parallel training on multiple GPUs
- using device var ensures training also works with no GPUs available
- Pytorch requires sending data to GPU memory explicitly
Communicating w Ray Tune
- send the validation loss and accuracy back to Ray Tune. Ray Tune can then use these metrics to decide which hyperparameter configuration lead to the best results
    - metrics can also be used to stop bad performing trials early in order to avoid wasting resources on those trials.
Test set accuracy
- hold-out test set with data that has not been used for training the model
Configuring speech space
- tune.choice() accepts a list of values that are uniformly sampled from
- l1 and l2 parameters should be powers of 2 between 4 and 256
- lr (learning rate) should be uniformly sampled between 0.0001 and 0.1
- batch size is a choice between 2, 4, 8, and 16
- at each trial Ray Tune randomly samples a combination of parameters from these search spaces
- ASHAScheduler terminates bad performing trials early
- after model training, find the best performing one and load the trained network from the checkpoint file. Then obtain the test set accuracy and report everything by printing.
# TensorBoard HParams Dashboard Guide
- HParams dashboard in TensorBoard provides several tools to help with this process of identifying the best experiment or most promising sets of hyperparameters
Experiment setup and the HParams experiment summary
- Three hyperparameters in model: # units in the first dense layer, dropout rate in the dropout layer, optimizer
- list values to try - ex. HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
Adapt TensorFlow runs to log hyperparameters and metrics
- model: two dense layers with a dropout layer between them
- for each run, log an hparams summary with the hyperparameters and final accuracy
Start runs and log them all under one parent directory
- use a grid search: try all combinations of the discrete parameters and just the lower and upper bounds of the real-valued parameter
Visualize the results in TensorBoard's HParams plugin
- start TensorBoard and click on "HParams" at the top
- left pane offers filtering of hyperparameters, metrics, run status, etc.
- dashboard has 3 diff views:
    - Table View lists the runs, their hyperparameters, and their metrics
    - Parallel Coordinates View shows each run as a line going through an axis for each hyperparemeter and metric
    - Scatter Plot View shows plots comparing each hyperparameter/metric with each metric, can help identify correlations
# Goodfellow et al., “Hyperparameter Search” (Chapter 11)
Practical Methodology
- need to know how to choose an algorithm for a particular application and how to monitor and respond to feedback obtained from experiments in order to improve a machine learning system
- Practical design process:
    - determine goals: error metric to use & target value. Should be driven by problem application seeks to solve
    - establish a working end to end pipeline asap, including estimation of performance metrics
    - instrument system well to determine bottlenecks in performance
    - repeatedly make incremental changes
11.1 Performance metrics
- error metric guides future actions
    - in an academic setting error rate derived from previously benchmarked results
- measuring with precision & recall: precision = fraction of detections reported by the model that were correct, recall = fraction of true events that were detected
    - common to plot PR curve, precision on y-axis & recall on x-axis
    - classifier generates a score that is higher if the event to be detected occurred
- summarize performance w single #: convert precisionpand recallrinto anF-score given by F = 2pr/p+r
- coverage: fraction of examplesfor which the machine learning system is able to produce a response
11.2 Default Baseline Models
- speech recognition should begin with an appropriate deep learning model
- supervised learning with ﬁxed-size vectors as input, use a feedforward network with fully connected layers
- input has known topological structure (ex., input is an image), use a convolutional network
    - begin by using some kind of piecewise linear unit
- input or output is a sequence, use a gated recurrent net (LSTM or GRU)
- reasonable opitmization algo = SGD with momentum witha decaying learning rate or Adam
- batch normalization can have a dramatic eﬀect on optimization performance
    - should be introduced quickly if optimization seems problematic
    - can reduce generalization error and allows dropout to be omitted
- unless dataset massive, should include some mild forms of regularization from the start
    - early stopping, dropout
- only use unsupervisedlearning in your ﬁrst attempt if the task you want to solve is unsupervised
11.3 Determining whether to gather more data
- often much better to gather more data than to improve the learning algorithm
- if performance on training set is poor, the learning algorithm is not using the training data that is already available, so there is no reason to gather more data
    - try increasing thesize of the model by adding more layers or adding more hidden units to each layer
    - try improving the learning algorithm, ex. by tuning the learningrate hyperparameter
- if large models & carefully tuned optimization algos don't work well - problem might be the quality of the training data
- If test set performance is much worse than training set performance, then gathering more data is one of the most eﬀective solutions
- simple alternative to gathering more data: reduce the size of the model or improvere gularization, by adjusting hyperparameters or by adding regularization strategies 
- gap between train and test performance is unacceptable even after tuning regularization hyperparameters - gathering more data is advisable
11.4 Selecting Hyperparameters
- 2 basic approaches to choosing these hyperparameters: choosing manually and choosing automatically
11.4.1 Manual Hyperparameter Tuning
- must understand the relationship betweenhyperparameters, training error, generalization error and computational resources(memory and runtime)
- goal usually to ﬁnd the lowest generalization error subject to some runtime and memory budget
- alt goal: adjust the eﬀective capacity of the model to match the complexity of the task
    - effective capacity constrained by 3 factors:
        - representational capacity of model
        - ability of learning algorithm to successfully minimize the cost function used to train the model
        - degree to which the cost function and training procedure regularize the model
    - model with more layers and more hidden units per layer has higher representational capacity - capable of representing more complicated functions
- generalization error typically follows a U-shaped curve when plotted as a function of one of the hyperparameters
    - somewhere in the middle lies the optimal model capacity, which achieves the lowest possible generalization error, by adding a medium generalization gap to a medium amount of training error
    - some hyperparameters can only subtract capacity
- learning rate most important - controls the eﬀective capacity of the model in a more complicated way than other hyperparameters
    - effective capacity highest when the learning rate is correct for the optimization problem
- tuning parameters other than learning rate requires monitoring both training and test error to diagnose whether model is overﬁtting or underﬁtting,then adjusting capacity appropriately
- error on the training set is higher than your target error rate: no choice but to increase capacity
- error on the test set is higher than your target error rate:
    - to reduce gap, change regularization hyperparameters to reduce eﬀective model capacity
- most hyperparameters can be set by reasoning about whether they increase or decrease model capacity
- do not lose sight of end goal: good performance on test set
    - adding regularization only way to achieve
    - brute force way: continually increase model capacity and training set sizeuntil the task is solved
        - increases computational cost of training and inference
Effect of Hyperparameters on Model Capacity
Number of hidden units
- Increases capacity when... increased
- Reason... increases the representational capacity of the model
- Caveats... increases both the time and memory cost of essentially every operation on the model
Learning rate
- Increases capacity when... tuned optimally
- Reason... improper learning rate, whether too high or too low, results in a model with low eﬀective capacity due to optimization failure
Convolutional Kernel Width
- Increases capacity when... increased
- Reason... increases # of parameters in the model
- Caveats... wider kernel results in a narrower output dimension, reducing model capacity unless you use implicit zero padding to reduce this eﬀect. Wider kernels require more memory for parameter storage and increase runtime, but a narrower output reduces memory cost
Implicit zero padding
- Increases capacity when... increased
- Reason... adding implicit zeros be-fore convolution keeps therepresentation size large
- Caveats... increases time and memory cost of most operations
Weight Decay coefficient
- Increases capacity when... decreased
- Reason... decreasing the weight decay coeﬃcient frees the model parameters to become larger
Dropout rate
- Increases capacity when... decreased
- Reason... Dropping units less oftengives the units more oppor-tunities to “conspire” witheach other to ﬁt the train-ing set
11.4.2 Automatic Hyperparameter Optimization Algorithms
- hyper parameter optimization algorithms: wrap a learning algorithm and choose its hyperparameters, thus hiding the hyperparameters of the learning algorithm from the user
- often have their own hyperparameters
    - usually easier to choose - acceptable performance may be achieved on a wide range of tasks using the same secondary hyperparameters for all tasks
11.4.3 Grid Search
- 3 or fewer hyperparameters
    - computational cost grows exponentially with the number of hyperparameters
- for each hyperparameter, the user selects a small ﬁnite set of values to explore
- grid search algorithm then trains a model for every joint speciﬁcation of hyperparameter values in the Cartesian product of the set of values for each individual hyperparameter
11.4.4 Random Search
- simple to program, moreconvenient to use, and converges much faster to good values than grid search
- first deﬁne a marginal distributionfor each hyperparameter
    - do not discretize or bin the values of the hyperparameters, so that we can explore a larger set of values and avoid additional computational cost
- faster bc no wasted experimental runs
11.4.5 Model-Based Hyperparameter Optimization
- search for good hyperparameters can be cast as an optimization problem
    - decision variables = hyperparameters, cost to be optimized = validation set error that results from training using these hyperparameters
- to compensate for lack of gradient of some diﬀerentiable error measure on the val set - can build a model of the val set error, then propose new hyperparameter guesses by performing optimization within this model
- optimization involves trade-off between between exploration and exploitation
    - exploration - proposing hyperparameters for that there is high uncertainty, which may lead to a large improvement but may also perform poorly
    - explotiation - proposing hyperparameters that the model is conﬁdent will perform as well as any hyperparameters it has seen so far—usually very similar to ones seen before
- common drawback - require for a training experiment to run to completion before they are able to extract any information from it (less efficient)
11.5 Debugging Strategies
- with poor machine learning performance, difficult to tell whether the poor performance is intrinsic to the algorithm itself or whether there is a bug in the implementation of the algorithm
- Debugging strats: design a case that is so simple that the correct behavior actually can be predicted, or we design a test that exercises one part of the neural net implementation in isolation
Debugging tests:
- Visualize the model in action:
    - when training a model, produce visualizations of the output
    - Directly observing the machine learning model performing its task will help to determine whether the quantitative performance numbers it achieves seem reasonable
- Visualize the worst mistakes
    - By viewing the training set examples that are the hardest to model correctly, one can often discover problems with the way the data have been preprocessed or labeled
- Reason about software using training and test error
    - If training error is low but test error is high, likely that that the training procedure works correctly, and the model is overﬁtting for fundamental algorithmic reasons
        - alternatively, test error is measured incorrectly because of a problem with saving the model after training then reloading, or because the test data was prepared diﬀerently from the training data
- Fit a tiny dataset
    - usually even small models can be guaranteed to be able ﬁt a suﬃciently small dataset
    - if you cannot train a classiﬁer to correctly label a single example, an autoencoder to successfully reproduce a single example with high ﬁdelity, or a generative model to consistently emit samples resembling a single example, there is a software defect preventing successful optimization on the training set
- Compare back-propagated derivatives to numerical derivatives
    - common source of error is implementing gradient expression incorrectly
    - verify by comparing the derivatives computed by your implementation of automatic diﬀerentiation to the derivatives computed by ﬁnite diﬀerences
    - can improve the accuracy of the approximation by using the centered diﬀerence
- Monitor histograms of activations and gradient
    - often useful to visualize statistics of neural network activations and gradients, collected over a large amount of training iterations
    - preactivation value of hidden units can tell us if the units saturate, or how often they do
    - useful to compare themagnitude of parameter gradients to the magnitude of the parameters themselves
        - would like the magnitude of parameter updates over a minibatch to represent something like 1 percent of the magnitude of the parameter
- many deep learning algorithms provide some sort of guarantee aboutthe results produced at each step
    - can be debugged by testing each of their guarantees
11.6 Example: Multi-digit Number recognition
- process began with data collection
- transcription task was preceded by a signiﬁcant amount of dataset curation, including using other machine learning techniques to detect the house numbers prior to transcribing them
- important general principle is to tailor the choice of metric to the business goals for the project
- after choosing quantitative goals, next step is to rapidly establish a sensible baseline system
    - iteratively reﬁne the baseline and test whether each change makes an improvement
- instrumenting the training and test set performance to determine whether the problem was underﬁtting or overﬁtting
    - debugging by visualizing the model’s worst errors
        - visualizing the incorrect training set transcriptions that the model gave the highest conﬁdence
- last performance percentage points came from adjusting hyperparameters - making the model larger while maintaining some restrictions on its computational cost
# Weights & Biases Sweeps Tutorial
- W&B Sweeps to create an organized and efficient way to automatically search through combinations of hyperparameter values
- 3 simple steps to running:
    - define the sweep: creating a dictionary or a YAML file that specifies the parameters to search through, the search strategy, the optimization metric et all
    - initialize the sweep: with one line of code we initialize the sweep and pass in the dictionary of sweep configurations: sweep_id = wandb.sweep(sweep_config)
    - run the sweep agent: call wandb.agent() and pass the sweep_id to run, along with a function that defines your model architecture and trains it: wandb.agent(sweep_id, function=train)
- Before starting:
    - pip install wandb -Uq, import wandb, wandb.login()
1. Define the sweep
    - must be in a nested dictionary if you start a sweep in a Jupyter Notebook. If you run a sweep within the command line, you must specify your sweep config with a YAML file
    Pick a search method
        - specify a hyperparameter search method within your configuration dictionary: grid, random, Bayesian search
        - specify a metric that you want to optimize for
    Specify hyperparameters to search through
        - specify one or more hyperparameter names to the parameter key and specify one or more hyperparameter values for the value key
        - values you search through for a given hyperparamter depend on the type of hyperparameter you are investigating
        - track a parameter but not vary its value: add the hyperparameter to your sweep configuration and specify the exact value that you want to use
2. Initialize the sweep
    - W&B uses a Sweep Controller to manage sweeps on the cloud or locally across one or more machines
    - component that actually executes a sweep = sweep agent. Activated on local machine
    - activate a sweep controller with the wandb.sweep method:
        sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
        - returns a sweep_id that you will use at a later step to activate your sweep
3. Define your machine learning code
    - before executing sweep, define the training procedure that uses the hyperparameter values you want to try
    - key to integrating W&B Sweeps into your training code is to ensure that, for each training experiment, training logic can access the hyperparameter values you defined in your sweep configuration
    W&B Python SDK methods in train:
        - wandb.init(): Initialize a new W&B run. Each run is a single execution of the training function
        - wandb.config: Pass sweep configuration with the hyperparameters you want to experiment with
        - wandb.log(): Log the training loss for each epoch
4. Activate sweep agents
    - responsible for running an experiment with a set of hyperparameter values that you defined in your sweep configuration
    - Create sweep agents with the wandb.agent method. Provide the following:
        - sweep the agent is a part of (sweep_id)
        - function the sweep is supposed to run
        - (optionally) How many configs to ask the sweep controller for (count)
Visualize sweep results
- Parallel Coordinates Plot
    - maps hyperparameter values to model metrics
    - useful for honing in on combinations of hyperparameters that led to the best model performance
- Hyperparameter Importance Plot
    - surfaces which hyperparameters were the best predictors of your metrics
    - report feature importance (from a random forest model) and correlation (implicitly a linear model)