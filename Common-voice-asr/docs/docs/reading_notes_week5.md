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
        mean and variance statistics are accumulated over a single time-step of the minibatch. Does not lead to improvements in optimization
Sequence-wise normalization → overcomes issues, for each hidden unit, we compute the mean and variance statistics over all items in the minibatch over the length of the sequence
BatchNorm difficult to implement for a deployed ASR system, since it is often necessary to evaluate a single utterance in deployment rather than a batch
store a running average of the mean and variance for the neuron collected during training, and use these for evaluation in deployment → can evaluate a single utterance at a time with better results than evaluating with a large batch















