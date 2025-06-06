# PyTorch “Writing Custom Datasets, DataLoaders & Transforms”
- custom dataset inherits Dataset & overrides __len__ & __get_item__ 
- Dataloader provides features to batch, shuffle, & load data
    - collate_fn specifies how samples need to be batched - customizable

# librosa documentation for melspectrogram and power_to_db
melspectrogram
- librosa.feature.melspectrogram(*, y=None, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant', power=2.0, **kwargs)
power_to_db
- compute a mel-scaled spectrogram
- librosa.power_to_db(S, *, ref=1.0, amin=1e-10, top_db=80.0)
- convert a power spectrogram (amplitude squared) to decibel (dB) units

# PyTorch tutorial “Speech Command Classification with torchaudio”
- Use torchaudio to download and represent the dataset -> converts audio files to tensors
- torchaudio.load() - returns a tuple containing the newly created tensor along with the sampling frequency of the audio file 
- class model(nn.Module): __init__ & forward

# PyTorch “TensorBoard with PyTorch” quick-start
- create Summary writer instance:
    - writer = SummaryWriter()
- to log a scalar value, use add_scalar(tag, scalar_value, global_step=None, walltime=None)
- start tensorboard : tensorboard --logdir=runs
- do not need the summary writer anymore, call close() method

# Deep Learning (Goodfellow et al.) – Chapter 6 (re-read/skim sections 6.1–6.3) and Chapter 10 (section 10.2)
6.1
- Use nonlinear function to describe features → done by using affine transformation controlled by learned parameters, followed by a fixed nonlinear function → activation function
    - Default recommendation to use rectified linear unit → activation function g(z) = max{0, z}
6.2
- Nonlinearity of neural network causes most interesting loss functions to become nonconvex → usually trained by using iterative, gradient-based optimizers that drive cost function very low
    - Stochastic gradient descent applied to nonconvex loss functions has no convergence guarantee & is sensitive to values of initial parameters
        - Important to initialize all weights to small random vals
    - Training algo based on using gradient to descend the cost function
6.2.1
- Use principle of maximum likelihood → cross-entropy between training data & model predictions as cost function
    - Total cost function to train NN often combines one of primary cost functions with a regularization term
6.2.1.1
- Cost function is log-likelihood (cross-entropy between training data & model distribution)
    - Removes burden of designing cost functions for each model
    - Gradient of cost function must be large & predictable enough to serve as a good guide for the learning algorithm
6.2.1.2
- Often want to learn just one conditional statistic of y given x
- For powerful NNs, can view cost function as being a functional → mapping from functions to real numbers
    - Learning as choosing a function rather than merely choosing a set of parameters
- Calculus of variations required for solving optimization problem for function, used to derive 2 results
    - If we could train on infinitely many samples from true data generating distribution, minimizing mean squared error cost function would give a function that predicts the mean of y for each value of x
    - Mean absolute error → function that predicts the median value of y for each x
        - MSE & MAE often lead to poor results when used w gradient-based optimization
6.2.2
- Choice of cost function tightly coupled w choice of output unit → determines form of cross-entropy function
- Any NN unit used as an output can be used as a hidden unit internally
- Role of output layer to provide some additional transformation from features to complete task tha network must perform
6.2.2.1
- Unit based on affine transformation w no nonlinearity
- Linear output layers often used to produce mean of a conditional Gaussian distribution
- Maximizing log-likelihood is equivalent to minimizing mean squared error
- Covariance must be constrained to a positive definite matrix for all inputs → difficult to satisfy constraints w a linear output layer so typically other units are used to parametrize covariance
- Do not saturate so pose little difficulty for gradient-based optimization algos
6.2.2.2
- Predicting the value of a binary variable y
    - Max likelihood approach to define Bernoulli distribution over y conditioned on x
    - Use different approach that ensure always a strong gradient when model answers wrong → based on using sigmoid output units combined w maximum likelihood
- Sigmoid unit w two components → uses linear layer to compute z = w^Th+b & ises sigmoid activation function to convert  into a probability - Sigmodal transformation z variable defining distribution over binary vars = logit
- Saturation only occurs when model has correct answer → when y = 1 & z is very positive or y = 0 & z is very negative
    - Gradient-based learning can act quickly to correct a mistaken z
- Max likelihood almost always preferred approach to training sigmoid output units
- Sigmoid returns values restricted to open interval (0,1) rather than [0,1]
6.2.2.3
- Wish to represent a probability distribution over a discrete variable with n possible values
- Often used as output of a classifier, to represent probability distribution over n different classes
- Unregularlized max likelihood will drive model to learn parameters that drive softmax to predict fraction of counts of each outcome observed in training set
- Softmax actuvatui cab saturate, has multiple output values than cat saturate when differences between input vals become extreme
    - When softmax saturates, cost functions based on softmax saturate unless they are able to invert the saturating activating function
- Can derive a numerically stable variant of softmax that enables evaluation w only small numerical errors
- Difficulties for learning if loss function not designed to compensate for output softmaz(z) saturation
- Argument z can be produced in two diff ways
    - Have an earlier layer of NN output every element of z
        - Overparametrizes distribution
    - Impose requirement that one element of z be fixed
- Softmax is continuous & differentiable
6.2.2.4
- NN can generalize to almost any kind of output layer, principle of max likelihood provides guide for designing good cost function for any kind
- NN represent a function → outputs not direct predictions of value y, provides parameters for a distribution over y
- Often want to perform multimodal regression → predict real values from a conditional distribution p(y | x) that can have several diff peaks in y space for the same val of x
    - Neural networks w Gaussian mixtures as output → mixture density networks
        - Must have three outputs that satisfy different constraints
        - A vector defining mixture components → form a  multinoulli distribution over n diff components associated w latent variable to guarantee outputs are positive & sum to 1
            - Means: indicate center or mean associated w i-th Gaussian component & are unrestrained. Wnat to update mean for component that actually produced the observation (unknown in practice)
            - Covariances: specify covariance matrix for each component i. Typically use diagonal matrix to avoid needing to compute determinants. Gradient descent automatically follows correct process if given correct specification of negative log-likelihood under mixture model
- Gaussian mixture outputs particularly effective in generative models of speech
6.3
- Predicting in advance what works best usually impossible → design process of trial & error with training network on kind of hidden unit thought to work well & evaluating performance w validation set
- Rectified linear units → excellent default choice
    - Not differentiable at z = 0, but does not invalidate use of gradient-based learning algo
        - Do not expect training to reach point where gradient is 0 so acceptable for minima of cost function to correspond to points w undefined gradient
- In practice one can safely disregard nondifferentiability of hidden unit activation functions
- Unless indicated otherwise, most hidden inputs described as accepting a vector of inputs x, computing an affine transformation z & applying an element-wise nonlinear function g(z)
6.3.1
- Use activation function g(z) = max{0,z}
- Easy to optimize bc so similar to linear units → difference is rectified outputting zero across half its domain
    - Makes derivatives remain large whenever unit is active
        - 2nd deriv of rectifying operation is 0 almost everywhere, derivative 1 everywhere unit is active
    - Gradients not only large but consistent 
- Drawback is they cannot learn via gradient-based methods on examples with activation 0
- Three generalizations based on using nonzero slope:
    - Absolute value rectification makes it -1 to make g(z) = |z|, used for object recognition
    - Leaky ReLU fixes slope to small value
    - Parametric ReLU traits slope as learnable parameter 
    - Maxout units generalize RLU further, divide z into groups of k values, each unit outputs max element of group
        - Provides a way of learning a piecewise linear function that responds to multiple directions in input x space
        - Can be seen as learning activation function itself
        - Typically need more regularization, but can work well without if training set large & # pieces per unit kept low
        - Unit driven by multiple filters, resistant to catastrophic forgetting → neural networks forget how to perform tasks they were trained on in the past
- Based on principle that models are easier to optimize if behavior is closer to linear
6.3.2
- Sigmoidal units
    - Saturate across most of domain → can make gradient-based learning very difficult
- Hyperbolic Tangic activation function typically performs better than logistic sigmoid
    - Resembles identity function more closely → training deep neural network resembles training linear network (as long as small activations)
- Not to have activation g(z) at app → identity function as activation function
- Linear unit as hidden unit
    - Offer effective way of reducing parameters in a network for small q outputs
- Softmax units → naturally represent prob distribution over a discrete var with k possible values, so may be used as a kind of switch
- Radial basis function → becomes more active as x approaches a template. Saturates to 0 for most x so difficult to optimize
- Softplus → smooth version of rectifier. Use generally discouraged
- Hard tanh → similar to tanh & rectifier but bounder
10.2
- Wide variety RNN's
    - produce an output at each time step and have recurrent connections between hidden units
    - produce an output at each time step and haverecurrent connections only from the output at one time step to the hiddenunits at the next time step
    - recurrent connections between hidden units, thatread an entire sequence and then produce a single output
- forward propagation equations for the RNN - assume hyperbolic tangent activation function & discrete output - as if the RNN is used to predict words or characters
- represent discrete variables is to regard the output o as giving the unnormalized log probabilities of each possible value of the discrete variable
- forward prop begins w specification of initial state h^(0). Then for each time step from t = 1 to t = r apply update equations
- total loss for a given sequence of x values paired with a sequence of y values would then be just the sum of the losses over all the time steps
- network with recurrence between hidden units very powerful but expensive to train
10.2.1
-  network with recurrent connections only from the output at one time step to the hidden units at the next time step -> lacks hidden-to-hidden recurrent connections -> requires that  output units capture all the information about the past that  network will use to predict future
    - unlikely to capture the necessary information about the past history of the input
- advantage -> for any loss function based on comparing the prediction at timet to the training target at time t, all the time steps are decoupled -> parallel training enabled
- Can be trained w teacher forcing ->  during training the model receives the ground truth output y^(t) as input at time t+ 1
    - allows avoiding back-propagationthrough time in models that lack hidden-to-hidden connections
- max likelihood specifies during training connections should be fed with the target values specifying what the correct output should be
- when hidden units become a function of earliertime steps, back prop through time algorithm is necessary & paired w teacher forcing
10.2.2
- computing gradient through a recurrent neural network -> apply generalized back-prop algo to unrolled computational graph
- once gradients on internal nodes of computational graph are obtained, can obtain gradients on parameter nodes
10.2.3
- loss should be chosen based on the task
    - feed-forward network -> use cross-entropy associated with that distribution
- use a predictive log-likelihood training objective -> train the RNN to estimate the conditional distribution of the next sequence element y^(t)given the past inputs
- feed the actual y values (not their prediction, but the actual observed or generated values)back into the network, the directed graphical model contains edges from ally(i)values in the past to the current y(t)value
- RNN as deﬁning a graphical model whose structure is the complete graph, able to represent direct dependencies between any pair of y values
- price RNNs pay for their reduced number of parameters is optimizing the parameters may be diﬃcult
- draw samples from model -> sample from the conditional distribution at each time step + needs mechanism for determining length of sequence
10.2.4
- RNNs allow the extension of the graphical model view to represent not only a joint distribution over y variables but also a conditional distribution overy given x
- rather than only single vector input, RNN can recieve sequence
# AssemblyAI blog “Word Error Rate (WER) explained”
- Word error rate -> calculates how many “errors” are in the transcription text produced by an ASR system, when compared to a human transcription
    - calculated: number of Substitutions (S), Deletions (D), and Insertions (N), divided by the Number of Words (N).
    - can tell you how “different” the automatic transcription was compared to the human transcription, and generally, reliable metric to determine how “good” an automatic transcription is.
    - can be unreliable -> not "smart", requires normalization by lowercasing,punctuation removal, numbers to written form to account for penalization
- important to choose datasets:
    - relevant to the intended use case of the ASR model
    - simulate real-world datasets by already having, or by adding, noise to the audio file
- consistent normalizer helps evaluators be more confident in the results of their comparison