# Attention is all you need
Transformer based solely on attention mechanisms, dispensing with recurrence and convolutions entirely
More parallelizable and requiring significantly less time to train
1 - Introduction
- fundamental constraint of sequential computation - parralelization 
- Attention mechanisms allow modeling of dependencies without regard to their distance in the input or output sequences
- Transformer relies on an attention mechanism to draw global dependencies between input and output
    - allows for significantly more parallelization and can reach a new state of the art in translation quality after limited training
2 - Background
- Tranformer operations required to relate signals from two arbitrary input or output positions is constant
- Self-attention = an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence
- End-to-end memory networks are based on a recurrent attention mechanism - language modeling tasks
3 -  Model Architecture
- Encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time
    - Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder
3.1 - Encoder & Decoder Stacks
- Encoder: a stack of N = 6 identical layers
    - Each layer has two sub-layers. First: multi-head self-attention mechanism, & second: simple, positionwise fully connected feed-forward network
        - Employ a residual connection around each of the two sub-layers, followed by layer normalization
            - Output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself
        - All sub-layers produce output dimension = 512
- Decoder: stack of N = 6 identical layers
    - Inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack
    - Modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions
    - Ensures that the predictions for position i can depend only on the known outputs at positions less than i.
3.2 - Attention
- Function can be described as mapping a query & a set of key-value pairs to an output, where the query, keys, values, & output are all vectors
- Output is computed as a weighted sum, weight assigned to each value is computed by a compatibility function of the query with the corresponding key
3.2.1 - Scaled Dot-Product Attention
- Input consists of queries and keys of dimension dk, & values of dimension dv
- Compute the dot products of the query with all keys, divide each by √dk, & apply a softmax function to obtain the weights on the values
- Compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V
    - Compute matrix of outputs: Attention(Q, K, V ) = softmax(QK^T/√dk)V
- Dot-product attention is identical to our algorithm, except for the scaling factor of 1/√dk
    - Much faster and more space-efficient in practice than additive attention
    - Counteract dot products growing large in magnitude with large values of dk, pushing the softmax function into regions where it has extremely small gradients through Scaled DPA
3.2.2 - Multi-Head Attention
- Beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively
- Perform the attention function in parallel on each projected versions of queries, yielding dv-dimensional output values
    - Outputs concatenated & then projected, resulting in final values
- Allows the model to jointly attend to information from different representation subspaces at different positions
- In this work employed h = 8 parallel attention layers & used dk = dv = dmodel/h = 64
3.2.3 - Applications of Attention in our Model
- Transformer uses multi-head attention in three different ways:
    - "encoder-decoder attention" layers, queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder
        - allows every position in the decoder to attend over all positions in the input sequence
    - Encoder contains self-attention layers
        - In a self-attention layer all of the keys, values and queries come from the same place
        - Each position in the encoder can attend to all positions in the previous layer of the encoder
    - Self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position
        - Need to prevent leftward information flow in the decoder to preserve the auto-regressive property
3.3 - Position-wise Feed-Forward Networks
- Each layer in encoder & decoder contains a fully connected feed-forward network, applied to each position separately and identically
    - Consists of two linear transformations with a ReLU activation in between
    - FFN(x) = max(0, xW1 + b1)W2 + b2
- Linear transformations are the same across different positions, but use different parameters from layer to layer
    - Ex. dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality df f = 2048
3.4 - Embeddings and Softmax
In their model:
- Use learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel
- Use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.
- Share the same weight matrix between the two embedding layers and the pre-softmax linear transformation
    - Multiply weights by √dmodel in embedding layers
3.5 - Positional Encoding
- Must inject some information about the relative or absolute position of the tokens in the sequence for model to make use of the order of the sequence
    - Add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks
        - same dimension dmodel as the embeddings, so that the two can be summed
        - Choices of positional encodings: learned & fixed
    - Each dimension of the positional encoding corresponds to a sinusoid
        - Sinusoidal may allow the model to extrapolate to sequence lengths longer than the ones encountered during training
4 - Why Self-Attention
- Computational complexity per layer
    - Self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d
        - most often the case with sentence representations
- Amount of computation that can be parallelized, measured by min number of sequential operations required
- Path length between long-range dependencies in the network
    - Learning long-range dependencies is a key challenge in many sequence transduction tasks
    - Factor affecting the ability to learn such dependencies is length of paths forward and backward signals have to traverse in the network – shorter the path, easier it is to learn long-range dependencies
        - Compare the maximum path length between any two input and output positions in networks composed of the different layer types
- Could yield more interpretable models
5.2 - Hardware & Scheduling 
- Trained the base models for a total of 100,000 steps or 12 hours
- Big models were trained for 300,000 steps (3.5 days).
5.3 - Optimizer
- Used the Adam optimizer with β1 = 0.9, β2 = 0.98 and ϵ = 10−9 . Varied the learning rate over the course of training, according to the formula:
    - lrate = dmodel-0.5* min(step_num-0.5, step_num * warmup_steps-1.5)
    - Corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number
    - Used warmup_steps = 4000
5.4 - Regularization
- Residual Dropout
    - apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized
    - apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks
    - base model, rate of Pdrop = 0.1
- Label Smoothing
    - employed label smoothing of value ϵls = 0.1
        - Hurts perplexity but improves accuracy
6.1 - Machine Translation
- Big transformer model outperforms prior reported models, at a fraction of the training cost 
6.2 - Model Variations
- Used beam search 
- Bigger models are better, and dropout is very helpful in avoiding over-fitting.
6.3 English Constituency Parsing
- Evaluating generalization to other tasks
- Increased the maximum output length to input length + 300. We used a beam size of 21 and α = 0.3
# Pytorch Transformer
- Architecture based on prior paper
- https://github.com/pytorch/examples/tree/main/word_language_model → apply nn.Transformer module for the word language model
- forward(...) → Take in and process masked source/target sequences. Returns a tensor
- https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer
# Deep Speech 2
End-to-end learning allows us to handle a diverse variety of speech including noisy environments, accents and different languages
1 Introduction
- Employ a spectrum of deep learning techniques: capturing large training sets, training larger models with high performance computing, and methodically exploring the space of neural network architectures
    - Able to reduce error rates & recognize Mandarin speech w high accuracy
    - Single engine must learn to be able to handle most applications with only minor modifications and able to learn new languages from scratch without dramatic changes
- Three crucial components: the model architecture, large labeled training datasets, and computational scale
- Neural networks trained with the Connectionist Temporal Classification (CTC) loss function to predict speech transcriptions from audio
- Deep learning systems benefit greatly from large quantities of training data → usually requires the use of larger models
2 Related Work
- Two methods are currently used to map variable length audio sequences directly to variable length transcriptions:
    - RNN encoder-decoder paradigm uses an encoder RNN to map the input to a fixed length vector and a decoder network to expand the fixed length vector into a sequence of output predictions 
    - Adding attentional mechanism to the decoder greatly improves performance of the system
3 Model Architecture
- Increase the model capacity via depth to learn from large datasets
    - Up to 11 layers including many bidirectional recurrent layers and convolutional layers
- Use Batch Normalization for RNNs and novel optimization curriculum SortaGrad
- Exploit long strides between RNN inputs to reduce computation per example by a factor of 3
3.1 Preliminaries
- Goal of the RNN is to convert an input sequence x into a final transcription y
- RNN makes a prediction over characters, p(lt|x), where lt is either a character in the alphabet or the blank symbol
- RNN model is composed of several layers of hidden units
    - Experimentations consist of one or more convolutional layers, followed by one or more recurrent layers, followed by one or more fully connected layers
- Use the clipped rectified linear (ReLU) function σ(x) = min{max{x, 0}, 20} as our nonlinearity
- Given an input-output pair (x, y) and the current parameters of the network θ, we compute the loss function L(x, y; θ) and its derivative with respect to the parameters of the network
    - derivative used to update the network parameters through the backpropagation through time algorithm
- Integrate a language model in a beam search decoding
3.2 Batch Normalization for Deep RNNs
- To efficiently scale models: increase the depth of the networks by adding more hidden layers, rather than making each layer larger
    - Explore Batch Normalization (BatchNorm) as a technique to accelerate training for such networks since they often suffer from optimization issues
- Two methods of extending BatchNorm to bidirectional RNNs:
    - transformation immediately before every non-linearity
        - mean and variance statistics are accumulated over a single time-step of the minibatch. Does not lead to improvements in optimization
    - Sequence-wise normalization → overcomes issues, for each hidden unit, we compute the mean and variance statistics over all items in the minibatch over the length of the sequence
- BatchNorm difficult to implement for a deployed ASR system, since it is often necessary to evaluate a single utterance in deployment rather than a batch
    - store a running average of the mean and variance for the neuron collected during training, and use these for evaluation in deployment → can evaluate a single utterance at a time with better results than evaluating with a large batch
3.3 SortaGrad
- CTC cost function that we use implicitly depends on the length of the utterance
    - longer examples tend to be more challenging
- SortaGrad uses the length of the utterance as a heuristic for difficulty, since long utterances have higher cost than short utterances.
- Find gains when applying SortaGrad and BatchNorm together
- Longer utterances are more likely to cause the internal state of the RNNs to explode at an early stage in training
3.4 Comparison of simple RNNs and GRUs
- Simple RNNs that have bidirectional recurrent layers with the recurrence for both the forward in time and backward in time directions
    - having a more complex recurrence can allow the network to remember state over more time-steps while making them more computationally expensive to train
- Common recurrent architectures: Long Short-Term Memory (LSTM) & Gated Recurrent Units (GRU)
    - Examined GRUs → faster to train and less likely to diverge
- Used clipped-ReLU as output nonlinearity for simplicity and uniformity with the rest of the network
- Both GRU and simple RNN architectures benefit from batch normalization and show strong results with deep networks
    - For a fixed number of parameters, the GRU architectures achieve better WER for all network depths
- GRU networks with 5 or more recurrent layers do not significantly improve performance
- Scale up the model size, for a fixed computational budget the simple RNN networks perform slightly better → opted for simple RNN
3.5 Frequency Convolutions
- Temporal convolution commonly used in speech recognition to efficiently model temporal translation invariance for variable length utterances
- Sub-sampling essential to make recurrent neural networks computationally tractable with high sample-rate audio
    - DS1 system accomplished this through the use of a spectrogram as input and temporal convolution in the first layer with a stride parameter to reduce the number of time-steps 
- Convolutions in frequency and time domains, applied to spectral input features prior to any other processing, can slightly improve ASR performance
    - Attempts to model spectral variance due to speaker variability more concisely than what is possible with large fully connected networks
    - Frequency convolutions work better as the first layers of the network
3.6 Striding
- Convolutional layers → apply a longer stride and wider context to speed up training as fewer time-steps are required to model a given utterance
- Downsampling the input sound reduces number of time-steps and computation required in the following layers, but at the expense of reduced performance
- In English, striding can reduce accuracy simply because the output of our network requires at least one timestep per output character, and the number of characters in English speech per time-step is high enough to cause problems
    - Overcome by enriching English alphabet with symbols representing alternate labellings → use non-overlapping bi-graphemes or bigrams that shorten output transcription length and allow for a decrease in unrolled RNN length
    - Bigrams allow for larger strides without any sacrifice in in the word error rate
3.7 Row Convolution and Unidirectional Models
- Unidirectional, forward-only RNN layers in our deployment system using row convolution → intuition that only need a small portion of future information to make an accurate prediction at the current time-step
    - Place the row convolution layer above all recurrent layers
        - allows us to stream all computation below the row convolution layer on a finer granularity given little future context is needed & results in better CER
    - Conjecture that the recurrent layers have learned good feature representations, so the row convolution layer simply gathers the appropriate information to feed to the classifier
3.8 Language Model
- WER improves when we supplement our system with a language model trained from external text
- Use an n-gram language model since they scale well to large amounts of unlabeled text 
- English language model is a Kneser-Ney smoothed 5-gram model with pruning that is trained using the KenLM toolkit
- During inference we search for the transcription y that maximizes Q(y) → linear combination of log probabilities from the CTC trained network and language model, along with a word insertion term
- Weight α controls the relative contributions of the language model and the CTC network. The weight β encourages more words in the transcription
    - Parameters are tuned on a development set, use a beam search to find the optimal transcription
3.9 Adaptation to Mandarin
- Performance of the beam search during decoding levels off at a smaller beam size. This allows us to use a beam size of 200 with a negligible degradation in CER
4 System Optimizations
- Built a highly optimized training system with two main components—a deep learning library written in C++, along with a high performance linear algebra library written in both CUDA and C++
    - Allows sustaining 24 single-precision teraFLOP/second when training a single model on one node
4.1 Scalability and Data-Parallelism
- Use standard technique of data-parallelism to train on multiple GPUs using synchronous Stochastic Gradient Descent (SGD)
    - Binds one process to each GPU, processes then exchange gradient matrices during the backpropagation by using all-reduce to exchange a matrix between multiple processes and sums the result so each process has a copy of the sum of all matrices from all processes at the end
    - Synchronous SGD reproducible & deterministic
- Time taken to train one epoch halves as we double the number of GPUs that we train on
- Wrote own implementation of the ring algorithm for higher performance and better stability, avoiding extraneous copies between CPU and GPU
4.2 GPU implementation of CTC loss function
- Originally, transferred activations from the GPUs to the CPU, calculating the loss function using an OpenMP parallelized implementation of CTC → limited scalability bc it became computationally more significant as RNN efficiency and scalability improved & transferring large activation matrices between CPU and GPU required spending interconnect bandwidth for CTC, rather than on transferring gradient matrices to allow scaling using data parallelism to more processors
- Parallel implementation relies on a slight refactoring to simplify the dependences in the CTC calculation+ use of optimized parallel sort implementations from ModernGPU 
4.3 Memory allocation
- Frequent use of dynamic memory allocations to GPU and CPU memory, mainly to store activation data for variable length utterances, and for intermediate results
- Wrote own memory allocator for both CPU and GPU allocations
    - Follows the approach of the last level shared allocator in jemalloc: all allocations are carved out of contiguous memory blocks using the buddy algorithm
    - To avoid fragmentation, preallocate all of GPU memory at the start of training and subdivide individual allocations from this block
    - Set the CPU memory block size that we forward to mmap to be substantially larger than std::malloc
- Most of the memory required for training deep recurrent networks is used to store activations through each layer for use by back propagation
- When a requested memory allocation exceeds available GPU memory, we allocate page-locked GPU-memory-mapped CPU memory using cudaMallocHost instead → can be accessed directly by the GPU by forwarding individual memory transactions over PCIe at reduced bandwidth & allows a model to continue to make progress even after encountering an outlier
5 Training Data
- English we use 11,940 hours of labeled speech data containing 8 million utterances 
5.1 Dataset Construction
- Developed an alignment, segmentation and filtering pipeline that can generate a training set with shorter utterances and few erroneous transcriptions
- Use an existing bidirectional RNN model trained with CTC to align the transcription to the frames of audio
    - Found CTC produces an accurate alignment when trained with a bidirectional RNN
- Segmentation step that splices the audio and the corresponding aligned transcription whenever it encounters a long series of consecutive blank labels occurs, usually denoting a stretch of silence
    - Require a space token to be within the stretch of blanks in order to segment only on word boundaries
- Removes erroneous examples that arise from a failed alignment
    - Word level edit distance between the ground truth and the aligned transcription is used to produce a good or bad label
- Filtering pipeline reduces the WER from 17% to 5% while retaining more than 50% of the examples
5.2 Data Augmentation
- Add noise to increase the effective size of our training data and to improve robustness to noisy speech
5.3 Scaling Data
- For each dataset, the model was trained for up to 20 epochs though usually early-stopped based on the error on a held out development set
- Speech system will continue to improve with more labeled training data
    - Hypothesize equally as important as increasing raw number of hours is increasing the number of speech contexts that are captured in the dataset (contest = any property that makes speech unique)
6 Results
- All models are trained for 20 epochs on either the full English dataset
- Use stochastic gradient descent with Nesterov momentum [61] along with a minibatch of 512 utterances
- Norm of the gradient exceeds a threshold of 400, it is rescaled to 400
- Model which performs the best on a held-out development set during training is chosen for evaluation
- Learning rate is chosen from [1 × 10−4 , 6 × 10−4 ] to yield fastest convergence and annealed by a constant factor of 1.2 after each epoch
- Beam size of 500 for the English decoder
6.1 English
- Best DS2 model has 11 layers with 3 layers of 2D convolution, 7 bidirectional recurrent layers, a fully-connected output layer along with Batch Normalization
- Read speech, accented speech, & noisy speech
6.1.1 Model Size
- English speech training set is substantially larger than the size of commonly used speech datasets
- To get the best generalization error, expect that the model size must increase to fully exploit the patterns in the data
7 Deployment
- System not well designed to transcribe in real time or with relatively low latency
    - Transcribing the first part of an utterance requires the entire utterance to be presented to the RNN
    - Use a wide beam when decoding with a language model so beam search can be expensive
        - Solve by modifying network and decoding procedure to produce a model that performs almost as well while having much lower latency
     - normalize power across an entire utterance, requires the entire utterance to be available in advance
        - Solve by using some statistics from our training set to perform an adaptive normalization of speech inputs during online transcription
7.1 Batch Dispatch
- Processing requests individually is inefficient computationally
    - processor must load all the weights of the network for each request → lowers the arithmetic intensity of the workload, and tends to make the computation memory bandwidth bound
    - Amount of parallelism that can be exploited to classify one request is limited
- RNNs are especially challenging to deploy → evaluating sample by sample relies on sequential matrix vector multiplications, which are bandwidth bound and difficult to parallelize
- Solve issues by building a batching scheduler called Batch Dispatch that assembles streams of data from user requests into batches before performing forward propagation on these batches
    - Still places constraints on the amount of batching we can perform
- Eager batching scheme that processes each batch as soon as the previous batch is completed, regardless of how much work is ready by that point → reduces end-user latency despite less computational efficiency
7.2 Deployment Optimized Matrix Multiply Kernels
- Using half-precision arithmetic saves memory space and bandwidth, which is especially useful for deployment
7.3 Beam Search
- Use a heuristic to further prune the beam search. Rather than considering all characters as viable additions to the beam, we only consider the fewest number of characters whose cumulative probability is at least p (p = 0.99 works well)
8 Conclusion
- Enhancements to numerical optimization through SortaGrad and Batch Normalization, evaluation of RNNs with larger strides with bigram outputs for English, searching through both bidirectional and unidirectional models
# Conformer
Transformer models are good at capturing content-based global interactions, while CNNs exploit local features effectively
Combine convolution neural networks and transformers to model both local and global dependencies of an audio sequence in a parameter-efficient way
1 Introduction
- Transformer architecture based on self-attention has enjoyed widespread adoption for modeling sequences due to its ability to capture long distance interactions and the high training efficiency
- Transformers are good at modeling long-range global context, they are less capable to extract finegrained local feature patterns
- Wu et al → multi-branch architecture with splitting the input into two branches: self-attention and convolution; and concatenating their outputs
- Organically combining convolutions with self-attention in ASR models → self-attention learns the global interaction whilst the convolutions efficiently capture the relative-offset-based local correlations
2 Conformer Encoder
- Audio encoder first processes the input with a convolution subsampling layer and then with a number of conformer blocks
- Conformer block is composed of four modules stacked together → a feed-forward module, a self-attention module, a convolution module, and a second feed-forward module in the end
2.1 Multi-Headed Self-Attention Module
- Employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL [20], the relative sinusoidal positional encoding scheme
    - Relative positional encoding allows the self-attention module to generalize better on different input length & encoder is more robust to utterance length variance
- Use pre-norm residual units with dropout which helps training and regularizing deeper models
2.2 Convolution Module
- Starts with a gating mechanism: pointwise convolution & GLU
- Followed by a single 1-D depthwise convolution layer
- Batchnorm deployed after convolution to aid training deep models
2.3 Feed Forward Module
- After the MHSA layer and is composed of two linear transformations and a nonlinear activation in between
- Residual connection is added over the feed-forward layers, followed by layer normalization
    - Structure is also adopted by Transformer ASR models
- Follow pre-norm residual units & apply layer normalization within the residual unit and on the input before the first linear layer + Swish activation & dropout, helps regularizing the network
2.4 Conformer Block
- Contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module and the Convolution module
- Employ half-step residual weights in our feed-forward (FFN) modules, the second feed-forward module is followed by a final layernorm layer
- Convolution module stacked after the self-attention module works best for speech recognition
3 Experiments
3.1. Data
- Use SpecAugment [27, 28] with mask parameter (F = 27), and ten time masks with maximum time-mask ratio (pS = 0.05), max size of time mask set to pS * utterance length
3.2 Conformer Transducer
- Sweeping different combinations of network depth, model dimensions, number of attention heads and choosing the best performing one within model parameter size constraints
- Use a single-LSTM-layer decoder in all models
- For regularization, apply dropout in each residual unit of the conformer, i.e, to the output of each module, before it is added to the module input. Rate of Pdrop = 0.1
- Train the models with the Adam optimizer with β1 = 0.9, β2 = 0.98 and  = 10-9 & a transformer learning rate schedule, with 10k warm-up steps and peak learning rate 0.05/ √d where d is the model dimension in conformer encoder
3.3 Results on LibriSpeech
- With language model added, achieves the lowest word error rate among all the existing models
3.4 Ablation Studies
3.4.1 Conformer Block vs. Transformer Block
- Conformer block includes a convolution block and has a pair of FFNs surrounding the block
- Advantage of placing the convolution module after the self-attention module in the Conformer block
- Sweep the kernel size in {3, 7, 17, 32, 65} of the large model → 32 best




