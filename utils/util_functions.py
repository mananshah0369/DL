DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

import inspect
import numpy as np
import tensorflow as tf
import sys
import re

nn_Module = tf.keras.Model

from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

import collections
from IPython import display

utils = sys.modules[__name__]

## Util Functions
def use_svg_display():
  backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
  use_svg_display()
  utils.plt.rcParams['figure.figsize'] = figsize

def add_to_class(Class):
  """Register functions as methods in created class"""
  def wrapper(obj):
    setattr(Class, obj.__name__, obj)
    return wrapper
  
## Util Classes
class HyperParameters:
  """Base Class of HyperParameters"""
  def save_hyperparameters(self, ignore=[]):
    frame = inspect.currentframe().f_back
    _, _, _, local_vars = inspect.getargvalues(frame=frame)
    self.hparams = {k: v for k, v in local_vars.items() if (k not in set(ignore+['self']) and not k.startswith('_'))}
    for k, v in self.hparams.items():
      setattr(self, k, v)

class ProgressBoard(utils.HyperParameters):
  """The board that plots data points in animation"""
  def __init__(self, xlabel=None, ylabel=None, xlim=None,  
               ylim=None, xscale='linear', yscale='linear',
               ls=['-', '--', '-.', ':'], 
               colors=['C0', 'C1', 'C2', 'C3'], 
               fig=None, axes=None, figsize=(6, 4), display=True):
    self.save_hyperparameters()

  def draw(self, x, y, label, every_n=1):
    Point = collections.namedtuple(typename='Point', field_names=['x', 'y'])
    if not hasattr(self, 'raw_points'):
      self.raw_points = collections.OrderedDict()
      self.data = collections.OrderedDict()

    if label not in self.raw_points:
      self.raw_points[label] = []
      self.data[label] = []

    points = self.raw_points[label]
    line = self.data[label]

    points.append(Point(x, y))
    if len(points) != every_n:
      return
  
    mean = lambda x: sum(x) / len(x)
    line.append(Point(mean(x=[p.x for p in points]),
                      mean(x=[p.y for p in points])))
    points.clear()

    utils.use_svg_display()
    if self.fig is None:
      self.fig = utils.plt.figure(figsize=self.figsize)

    plt_lines, labels = [], []
    for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
      plt_lines.append(utils.plt.plot([p.x for p in v], [p.y for p in v], linestyle=ls, color=color)[0])
      labels.append(k)

    axes = self.axes if self.axes else utils.plt.gca()
    if self.xlim: axes.set_xlim(self.xlim)
    if self.ylim: axes.set_ylim(self.ylim)
    if not self.xlabel: self.xlabel = 'x'
    axes.set_xlabel(self.xlabel)
    axes.set_ylabel(self.ylabel)
    axes.set_xscale(self.xscale)
    axes.set_yscale(self.yscale)
    axes.legend(plt_lines, labels)
    display.display(self.fig)
    display.clear_output(wait=True)

class DataModule(utils.HyperParameters):
  """Base class of data"""
  def __init__(self, root='../data'):
    self.save_hyperparameters()

  def get_dataloader(self, train):
    raise NotImplementedError
  
  def train_dataloader(self):
    return self.get_dataloader(train=True)
  
  def val_dataloader(self):
    return self.get_dataloader(train=False)
  
  def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    shuffle_buffer = tensors[0].shape[0] if train else 1
    return tf.data.Dataset.from_tensor_slices(tensors=tensors).shuffle(buffer_size=shuffle_buffer).batch(batch_size=self.batch_size)
  
class Module(utils.nn_Module, utils.HyperParameters):
  """Base Class for Models
     Methods to define
     a. init
     b. loss
     c. forward
  """
  def __init__(self, plot_train_per_epoch=1, plot_valid_per_epoch=1):
    super().__init__()
    self.save_hyperparameters()
    self.board = ProgressBoard()
    self.training = None

  def loss(self, y_hat, y):
    raise NotImplementedError
  
  def forward(self, X):
    assert hasattr(self, 'net'), 'Neural Network is defined'
    return self.net(X)

  def call(self, X, *args, **kwargs):
    if kwargs and "training" in kwargs:
      self.training = kwargs['training']
    return self.forward(X=X, *args)
  
  def plot(self, key, value, train):
    """Plot a point in animation"""
    assert hasattr(self, 'trainer'), 'Trainer is not inited'
    self.board.xlabel='epoch'
    
    if train:
      x = self.trainer.train_batch_idx / self.trainer.num_train_batches
      n = self.trainer.num_train_batches / self.plot_train_per_epoch
    else:
      x = self.trainer.epoch + 1
      n = self.trainer.num_val_batches / self.plot_valid_per_epoch

    self.board.draw(x=x, y=value.numpy(), label=('train_' if train else 'val_') + key, every_n=n)

  def training_step(self, batch):
    l = self.loss(y_hat=self(*batch[:-1]), y=batch[-1])
    self.plot(key='loss', value=l, train=True)
    return l
  
  def validation_step(self, batch):
    l = self.loss(y_hat=self(*batch[:-1]), y=batch[-1])
    self.plot(key='loss', value=l, train=False)

  def configure_optimizers(self):
    return tf.keras.optimizers.legacy.SGD(self.lr)
  
class Trainer(utils.HyperParameters):
  """Base Class for Training Models with Data"""
  def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    assert num_gpus == 0, 'No GPU support yet'

  def prepare_data(self, data):
    self.train_dataloader = data.train_dataloader()
    self.val_dataloader = data.val_dataloader()
    self.num_train_batches = len(self.train_dataloader)
    self.num_val_batches = len(self.val_dataloader if self.val_dataloader is not None else 0)

  def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    self.model = model

  def fit(self, model, data):
    self.prepare_data(data=data)
    self.prepare_model(model=model)
    self.optim = model.configure_optimizers()
    self.epoch = 0
    self.train_batch_idx = 0
    self.val_batch_idx = 0
    for self.epoch in range(self.max_epochs):
      self.fit_epoch()

  def prepare_batch(self, batch):
    return batch
  
  def fit_epoch(self):
    self.model.training = True
    for batch in self.train_dataloader:
      with tf.GradientTape() as tape:
        loss = self.model.training_step(self.prepare_batch(batch=batch))
        grads = tape.gradient(target=loss, sources=self.model.trainable_variables)
        if self.gradient_clip_val > 0:
          grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_batch_idx += 1

    if self.val_dataloader is None:
      return
    
    self.model.training = False
    for batch in self.val_dataloader:
      self.model.validation_step(self.prepare_batch(batch=batch))
      self.val_batch_idx += 1

  def clip_gradients(self, grad_clip_val, grads):
    """Defined in :numref:`sec_rnn-scratch`"""
    grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
    new_grads = [tf.convert_to_tensor(grad) if isinstance(
        grad, tf.IndexedSlices) else grad for grad in grads]
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
    if tf.greater(norm, grad_clip_val):
        for i, grad in enumerate(new_grads):
            new_grads[i] = grad * grad_clip_val / norm
        return new_grads
    return grads
  
class Classifier(utils.Module):
  def validation_step(self, batch):
    X_val, y_val = batch
    y_hat = self(X_val)
    l = self.loss(y_hat=y_hat, y=y_val)
    accuracy = self.accuracy(y_val, y_hat=y_hat)
    self.plot(key='loss', value=l, train=False)
    self.plot(key='accuracy', value=accuracy, train=False)

  def accuracy(self, y, y_hat):
    """
      y shape: (n_samples, )
      y_hat: (n_samples, n_classes) 
    """
    n_classes = y_hat.shape[-1]
    y_hat = tf.reshape(y_hat, (-1, n_classes))
    preds = tf.cast(tf.argmax(y_hat, axis=1), y.dtype)
    compare = tf.cast(preds == tf.reshape(y, (-1,)), tf.float32)
    return tf.reduce_mean(compare)

  def loss(self, y, y_hat, averaged=True):
    """
      y shape: (n_samples, )
      y_hat: (n_samples, n_classes) 
    """
    n_classes = y_hat.shape[-1]
    y_hat = tf.reshape(y_hat, (-1, n_classes))
    y = tf.reshape(y, (-1,))
    # y = tf.one_hot(y, depth=n_classes)
    # loss = -1*tf.reduce_mean(tf.math.log(tf.boolean_mask(y_hat, y)))
    reduction = 'sum_over_batch_size' if averaged else 'none'
    fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=reduction)
    loss = fn(y, y_hat)
    return loss

  def layer_summary(self, X_shape):
    X = tf.random.normal(shape=X_shape)
    for layer in self.net.layers:
      X = layer(X)
      print(layer.__class__.__name__, 'output shape:\t', X.shape)
  
class FashionMNISTData(utils.DataModule):
  def __init__(self, batch_size=64, resize=(32, 32)):
    super().__init__()
    self.save_hyperparameters()
    self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()
  
  def get_dataloader(self, train):
    data = self.train if train else self.val
    process = lambda X, y: (tf.expand_dims(X, axis=3)/255, tf.cast(y, dtype='int32'))
    target_height, target_width = self.resize
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, target_height=target_height, target_width=target_width), y)
    shuffle_buffer = data[0].shape[0] if train else 1
    X, y = process(*data)
    X, y = resize_fn(X, y)
    return tf.data.Dataset.from_tensor_slices((X, y)).batch(self.batch_size).shuffle(shuffle_buffer)
  
  def text_labels(self, indices):
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
  
  def show_images(self, X, y, n_rows=1, n_cols=8, preds=None):
    import matplotlib.pyplot as plt
    _, axs = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    axs = axs.flatten()
    imgs = X.numpy()[:n_cols]
    labels = self.text_labels(list(y.numpy()[:n_cols]))
    if preds is not None:
      preds = self.text_labels(list(preds.numpy()[:n_cols]))
    else:
      preds = [None]*len(labels)
    for img, ax, label, pred in zip(imgs, axs, labels, preds):
        ax.imshow(img)
        title = label
        if pred:
          title = label + "\n" + pred 
        ax.set_title(title)
        
    plt.show()

class Vocab:
  """Pass Raw Dataset splited on tokens or list(tokens) to vocab"""
  def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
    if tokens and isinstance(tokens[0], list):
      tokens = [token for line in tokens for token in line]
    counter = collections.Counter(tokens)
    self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    self.idx_to_token = list(sorted(['<unk>'] + reserved_tokens + \
                               [token for token, freq in self.token_freqs if freq >= min_freq]))
    self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

  def __len__(self):
    return len(self.idx_to_token)
  
  def __getitem__(self, tokens): ## Overload of [] operator
    if not isinstance(tokens, (list, tuple)):
      return self.token_to_idx.get(tokens, self.unk)
    return [self.__getitem__(token) for token in tokens]
  
  def to_tokens(self, indices):
    if hasattr(indices, '__len__') and len(indices) > 1:
      return [self.idx_to_token[idx] for idx in indices]
    return self.idx_to_token[indices]

  @property
  def unk(self):
    return self.token_to_idx['<unk>'] 

class TimeMachine(utils.DataModule):
  """Downloads the text. Tokenize the text. Pass it to vocab and get corpus and vocab as output. Generate X and y"""
  def _download(self):
    fname = "/Users/mananshah/DL/data/timemachine.txt"
    with open(fname) as f:
      return f.read()
    
  def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()
  
  def _tokenize(self, text):
    return list(text)
  
  def build(self, raw_text, vocab=None):
    tokens = self._tokenize(self._preprocess(raw_text))
    if vocab is None:
      vocab = Vocab(tokens=tokens)
      corpus = [vocab[token] for token in tokens]
      return corpus, vocab
    
  def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
    super().__init__()
    self.save_hyperparameters()
    corpus, self.vocab = self.build(self._download())
    array = tf.constant([corpus[i : i+num_steps+1] for i in range(len(corpus) - num_steps)])
    self.X, self.Y = array[:, :-1], array[:, 1:]

  def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(tensors=(self.X, self.Y), train=train, indices=i)
  
data = TimeMachine(batch_size=2, num_steps=10)

class RNN(utils.Module):
  ## Accepts One Hot Encoded Inputs
  ## Gives out list of states and last state
  def __init__(self, num_inputs, num_hiddens, sigma=0.1):
    super().__init__()
    self.save_hyperparameters()
    self.rnn = tf.keras.layers.SimpleRNN(units=num_hiddens, return_sequences=True, return_state=True)

  def forward(self, X, state=None):
    """
    X: (batch_size, num_steps, num_inputs)
    state: (num_batches, num_hiddens)
    """
    outputs, state = self.rnn(X, state)
    
    ## Outputs: shape (batch_size, num_steps, num_hiddens)
    ## State: shape (batch_size, num_hiddens)
    return outputs, state

class RNNLM(utils.Classifier):
  """Accepts RNN and Vocab as Input
     1. Encodes Input to One Hot
     2. Forward Method: Encodes input to one hot, Pass it on to rnn get the states, pass it to output layer to get logits
     3. Compute Loss
     4. Define Predict Method
  """
  def __init__(self, rnn, vocab_size, lr=0.01):
    super().__init__()
    self.save_hyperparameters()
    self.init_params()

  def init_params(self):
    self.W_hq = tf.Variable(initial_value=tf.random.normal(shape=(self.rnn.num_hiddens, self.vocab_size), mean=0, stddev=self.rnn.sigma))
    self.b_q = tf.Variable(tf.zeros(self.vocab_size))

  def one_hot(self, X):
    """X has shape: (batch_size, num_steps)
       Output: (batch_size, num_steps, vocab_size)
    """
    X_t = X
    return tf.one_hot(indices=X_t, depth=self.vocab_size, axis=-1)
  
  def output_layer(self, rnn_outputs):
    ## rnn_outputs: shape (batch_size, num_steps, num_hiddens)
    
    outputs = tf.matmul(rnn_outputs, self.W_hq) + self.b_q
    ## Shape (batch_size, num_steps, vocab_size)) 

    return outputs
  
  def forward(self, X, state=None):
    embs = self.one_hot(X)
    rnn_outputs, _ = self.rnn(X=embs, state=state)
    return self.output_layer(rnn_outputs=rnn_outputs)
    
  def training_step(self, batch):
    X, y = batch
    y_pred = self(X)
    l = self.loss(y=y, y_hat=y_pred)
    self.plot('ppl', tf.exp(l), train=True)
    return l

  def validation_step(self, batch):
    X, y = batch
    y_pred = self(X)
    l = self.loss(y=y, y_hat=y_pred)
    self.plot('ppl', tf.exp(l), train=False)

  def predict(self, prefix, num_preds, vocab, device=None):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
      X = tf.constant([[outputs[-1]]]) ## (num_batches, num_inputs)
      embs = self.one_hot(X)
      rnn_outputs, state = self.rnn(X=embs, state=state)
      if i < len(prefix) - 1:
        outputs.append(vocab[prefix[i+1]])
      else:
        Y = self.output_layer(rnn_outputs=rnn_outputs)
        output = int(tf.reshape(tf.argmax(Y, axis=2), shape=1))
        outputs.append(output)
    
    return ''.join(vocab.idx_to_token[i] for i in outputs)

class DeepRNN(utils.RNN):
  def __init__(self, num_hiddens, num_layers, dropout=0):
    utils.Module.__init__(self)
    self.save_hyperparameters()
    gru_cells = tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(units=num_hiddens, dropout=dropout) for _ in range(num_layers)])
    self.rnn = tf.keras.layers.RNN(gru_cells, return_sequences=True, return_state=True)

  def forward(self, X, state=None):
    ## Output (batch_size, timesteps, output_size), [(batch_size, output_size)]
    outputs, *state = self.rnn(X, state)
    return outputs, state
    
## Machine Translation Dataset
class MTFraEng(utils.DataModule):
  def _download(self):
    """Download File"""
    with open("../data/fra-eng/fra.txt", encoding='utf-8') as f:
      return f.read()
  
  def _preprocess(self, raw_text):
    """Remove non breaking spaces, add space before special character"""
    text = raw_text.replace('\u202f', ' ').replace('\xa0' , ' ')
    text.lower()
    no_space = lambda char, prev_char: char in ',?.!' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char 
           for i, char in enumerate(text.lower())]
    return ''.join(out)    
  
  def _get_src_tgt_arrays(self):
    raw_text = self._download()
    text = self._preprocess(raw_text)
    examples = text.split('\n')
    self.num_samples = len(examples)
    eng, fra = [], []
    for example in examples:
      example = example.split('\t')
      if (len(example) == 2):
        eng.append(example[0])
        fra.append(example[1])

    return eng, fra

  def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
    super().__init__()
    self.save_hyperparameters()
    
  def build(self, src_examples, tgt_examples, train=True):
    src_tokens = []
    for src_example in src_examples:
      # Add <eos> at end of each example
      # Pad sentences to length of num_steps using <pad> token
      # Convert each example into list of tokens
      src_example = src_example.split(' ') + ['<eos>']
      l = len(src_example)
      src_example = src_example[:self.num_steps] if l > self.num_steps else src_example + ['<pad>'] * (self.num_steps - l)
      src_tokens.append(src_example)

    tgt_tokens = []
    for tgt_example in tgt_examples:
      # Add <eos> at end of each example
      # Add <bos> at end of each example
      # Pad sentences to length of num_steps using <pad> token
      # Convert each example into list of tokens
      tgt_example =  ['<bos>'] + tgt_example.split(' ') + ['<eos>']
      l = len(tgt_example)
      tgt_example = tgt_example[:self.num_steps] if l > self.num_steps else tgt_example + ['<pad>'] * (self.num_steps - l)
      tgt_tokens.append(tgt_example)

    if train:
      self.src_vocab = utils.Vocab(tokens=src_tokens, min_freq=2)
      self.tgt_vocab = utils.Vocab(tokens=tgt_tokens, min_freq=2)
    
    self.src_array = tf.constant([self.src_vocab[s] for s in src_tokens])
    self.src_valid_len = tf.reduce_sum(input_tensor=tf.cast(
      x=self.src_array != self.src_vocab['<pad>'], dtype=tf.int32), 
      axis=1
    )
    self.tgt_array = tf.constant([self.tgt_vocab[s] for s in tgt_tokens])
    self.arrays = (self.src_array, self.tgt_array[:, :-1], self.src_valid_len, self.tgt_array[:, 1:])
    return self.arrays
  
  def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(tensors=self.arrays, train=train, indices=i)

## Encoder Decoder Architecture - To be fixed
class Encoder(tf.keras.layers.Layer):
  """Base Encoder Interface for encoder--decoder architecture"""
  def __init__(self):
    super().__init__()

  def call(self, X, *args):
    """
    X: shape (num_steps, batch_size)
    Output: (num_steps, batch_size, num_hidden), state
    """
    raise NotImplementedError
  
class Seq2SeqEncoder(Encoder):
  def __init__(self, vocab_size, embed_size, num_hiddens, num_layers=2, dropout=0):
    super().__init__()
    
    ## accepts (num_steps, batch_size) as input
    ## gives out (num_steps, batch_size, embed_size)
    self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size)

    ## Accepts (num_steps, batch_size, embed_size)
    ## gives out (num_steps, batch_size, num_hiddens), [(batch_size, num_hiddens)]
    self.rnn = utils.DeepRNN(num_hiddens=num_hiddens, num_layers=2, dropout=dropout)

  def call(self, X, *args):
    ## Accepts (batch_size, num_steps)
    ## gives out (num_steps, batch_size, num_hiddens), [(batch_size, num_hiddens)]

    encoder_inputs = self.embedding(tf.transpose(X))
    encoder_outputs, states = self.rnn(X=encoder_inputs, state=None)
    return encoder_outputs, states

class Decoder(tf.keras.layers.Layer):
  """Base Decoder Interface for encoder--decoder architecture"""
  def __init__(self):
    super().__init__()

  def init_state(self, enc_all_outputs, *args):
    """returns encoder outs"""
    raise NotImplementedError
  
  def call(self, X, state):
    """
    X: shape (batch_size, num_steps)
    state: from encoder encoder outputs
    Output: (num_steps, batch_size, vocab_size)
    """
    raise NotImplementedError
  
class Seq2SeqDecoder(tf.keras.layers.Layer):
  """Base Decoder Interface for encoder--decoder architecture"""
  def __init__(self, vocab_size, embed_size, num_hiddens, num_layers=2, dropout=0):
    super().__init__()
    
    ## accepts (num_steps, batch_size) as input
    ## gives out (num_steps, batch_size, embed_size)
    self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size)

    ## Accepts (num_steps, batch_size, embed_size)
    ## gives out (num_steps, batch_size, num_hiddens), [(batch_size, num_hiddens)]
    self.rnn = utils.DeepRNN(num_hiddens=num_hiddens, num_layers=2, dropout=dropout)
    self.dense = tf.keras.layers.Dense(units=vocab_size)
    
  def init_state(self, enc_all_outputs, *args):
    """returns encoder outs"""
    return enc_all_outputs
  
  def call(self, X, state):
    """
    X: shape (batch_size, num_steps)
    state: from encoder encoder outputs
    Output: (num_steps, batch_size, vocab_size)
    """
    num_steps = X.shape[1]
    decoder_inputs = self.embedding(tf.transpose(X))
    encoder_outputs, hidden_state = state
    
    context_vector = encoder_outputs[-1] 
    #(batch_size, num_hiddens)

    context_vector = tf.expand_dims(input=context_vector, axis=0) 
    # (1, batch_size, num_hiddens)
    
    context_vector = tf.tile(input=context_vector, multiples=[num_steps, 1, 1]) 
    # (num_steps, batch_size, num_hiddens)

    decoder_inputs = tf.concat([decoder_inputs, context_vector], axis=-1)

    decoder_outputs, hidden_state = self.rnn(X=decoder_inputs, state=hidden_state)  
    # (num_steps, batch_size, num_hiddens)
    
    outputs = self.dense(decoder_outputs) 
    # (num_steps, batch_size, vocab_size)

    outputs = tf.transpose(outputs, perm=[1, 0, 2]) 
    # (batch_size, num_steps, vocab_size)

    return outputs, [encoder_outputs, hidden_state]

class EncoderDecoder(utils.Classifier):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder

  def call(self, enc_X, dec_X, *args):
    """
      enc_X: Encoder Inputs (batch_size, num_steps)
      dec_X: Decoder Inputs (batch_size, num_steps-1)
    """
    enc_all_outputs = self.encoder(enc_X, *args, training=True)
    dec_state = self.decoder.init_state(enc_all_outputs, *args)
    return self.decoder(dec_X, dec_state, training=True)[0]
  
class Seq2SeqEncoderDecoder(EncoderDecoder):
  def __init__(self, encoder, decoder, tgt_pad, lr):
    super().__init__(encoder, decoder)
    self.save_hyperparameters()
    
  def call(self, enc_X, dec_X, *args):
    enc_all_outputs = self.encoder(enc_X, *args, training=True)
    dec_state = self.decoder.init_state(enc_all_outputs, *args)
    return self.decoder(dec_X, dec_state, training=True)[0]
  
  def loss(self, y_hat, y):
    loss = super().loss(y=y, y_hat=y_hat, averaged=False)
    mask = tf.cast(tf.reshape(y, shape=loss.shape) != self.tgt_pad, dtype=tf.float32)
    loss = tf.reduce_sum(mask * loss) / tf.reduce_sum(mask)
    return loss
  
  def validation_step(self, batch):
    y_hat = self(*batch[:-1])
    y = batch[-1]
    loss = super().loss(y=y, y_hat=y_hat, averaged=False)
    mask = tf.cast(tf.reshape(y, shape=loss.shape) != self.tgt_pad, dtype=tf.float32)
    loss = tf.reduce_sum(mask * loss) / tf.reduce_sum(mask)
    self.plot('loss', loss, train=False)

  def configure_optimizers(self):
    return tf.keras.optimizers.legacy.Adam(self.lr)
  
  def predict(self, batch, tgt_vocab, num_steps, save_attention_weights=False):
    arrays = data.build(src_examples=batch[0], tgt_examples=batch[1], train=False)
    src, tgt, valid_len, label = arrays
    enc_all_outputs = self.encoder(X=src)
    dec_state = self.decoder.init_state(enc_all_outputs, valid_len)
    outputs = [tf.expand_dims(tgt[:, 0], 1)]
    attention_weights = []
    for _ in range(num_steps):
      dec_output, dec_state = self.decoder(outputs[-1], dec_state, training=False)
      outputs.append(tf.argmax(dec_output, axis=2))
      if save_attention_weights:
        attention_weights.append(self.decoder.attention_weights)
        
    return tf.concat(outputs[1:], axis=1), attention_weights

## Attention Mechanism
def masked_softmax(X, valid_lens, value=0):
  """
  X: (batch_size, num_queries_per_sample, num_keys)
  valid_lens: (batch_size, ) OR (batch_size, num_queries_per_sample)
  """
  if valid_lens is None:
    return tf.nn.softmax(logits=X, axis=-1)
  
  shape = X.shape
  batch_size = shape[0]
  num_queries_per_sample = shape[1]
  num_keys = shape[2]

  X = tf.reshape(X, shape=(-1, num_keys)) 
  ## Shape: (batch_size * num_queries_per_sample, num_keys)
  if len(valid_lens.shape) == 1:
    valid_lens = tf.repeat(valid_lens, repeats=shape[1]) 
    ## Shape (batch_size * num_queries_per_sample, )
  else:
    valid_lens = tf.reshape(valid_lens, shape=-1)       
    ## Shape (batch_size * num_queries_per_sample, )

  mask = tf.range(0, num_keys)      ## Shape (num_keys, )
  mask = tf.expand_dims(mask, axis=0)     ## Shape (1, num_keys)
  mask = tf.tile(mask, multiples=[batch_size*num_queries_per_sample, 1])     
  ## Shape (batch_size * num_queries_per_sample, num_keys)

  valid_lens = tf.expand_dims(valid_lens, axis=1) 
  ## Shape (batch_size * num_queries_per_sample, 1)

  valid_lens = tf.tile(valid_lens, multiples=[1, num_keys])     
  ## Shape (batch_size * num_queries_per_sample, num_keys)

  masked_X = tf.where(mask < valid_lens, x=X, y=value)
  masked_X = tf.reshape(masked_X, shape=shape)
  return tf.nn.softmax(masked_X, axis=-1)

class DotProductAttention(tf.keras.layers.Layer):
  def __init__(self, dropout):
    super().__init__()
    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, queries, keys, values, valid_lens=None, **kwargs):
    """
    Queries: Decoder Input: (batch_size, num_queries_per_sample, dims)
    Keys: Encoder Hidden States:   (batch_size, num_keys, dims)         ## (num_keys = num_steps)
    Values: Encoder Hidden States: (batch_size, num_keys, dims)         ## (num_keys = num_steps)
    Valid Lens: (batch_size, ) OR (batch_size, num_queries_per_sample)  
    """
    dims = queries.shape[-1]
    scores = tf.matmul(a=queries, b=keys, transpose_b=True)/tf.math.sqrt(x=tf.cast(dims, dtype=tf.float32))
    ## (batch_size, num_queries, num_keys)
    self.attention_weights = masked_softmax(X=scores, valid_lens=valid_lens)
    ## (batch_size, num_queries, num_keys)

    weights = self.dropout(self.attention_weights, **kwargs) 
    ## (batch_size, num_queries, num_keys)

    ## (batch_size, num_queries, dims)
    return tf.matmul(weights, values)
  
class AdditiveAttention(tf.keras.layers.Layer):
  def __init__(self, key_dims, query_dims, num_hiddens, dropout, **kwargs):
    super().__init__(**kwargs)
    self.dropout = tf.keras.layers.Dropout(dropout)
    self.W_k = tf.keras.layers.Dense(units=num_hiddens, use_bias=False)
    self.W_q = tf.keras.layers.Dense(units=num_hiddens, use_bias=False)
    self.w_v = tf.keras.layers.Dense(units=1, use_bias=False)

  def call(self, queries, keys, values, valid_lens=None, **kwargs):
    """
    Queries: Decoder Input: (batch_size, num_queries_per_sample, dims_1)
    Keys: Encoder Hidden States: (batch_size, num_keys, dims_1)           ## (num_keys = num_steps)
    Values: Encoder Hidden States: (batch_size, num_keys, dims)           ## (num_keys = num_steps)
    Valid Lens: (batch_size, ) OR (batch_size, num_queries_per_sample)  
    """

    queries = self.W_q(queries) # (batch_size, num_queries_per_sample, num_hiddens)
    keys = self.W_k(keys)       # (batch_size, num_keys, num_hiddens)

    features = tf.expand_dims(input=queries, axis=2) + tf.expand_dims(input=keys, axis=1)
                # (batch_size, num_queries_per_sample, 1, num_hiddens) 
                # (batch_size, 1, num_keys, num_hiddens)
                # Output: (batch_size, num_queries_per_sample, num_keys, num_hiddens)
    features = tf.nn.tanh(features)
    scores = tf.squeeze(self.w_v(features), axis=-1)
    ## (batch_size, num_queries, num_keys)
    
    self.attention_weights = masked_softmax(X=scores, valid_lens=valid_lens)
    ## (batch_size, num_queries, num_keys)

    weights = self.dropout(self.attention_weights, **kwargs) 
    ## (batch_size, num_queries, num_keys)

    ## (batch_size, num_queries, dims)
    return tf.matmul(weights, values) 

class MultiHeadAttention(utils.Module):
  def  __init__(self, key_dims, query_dims, value_dims, num_hiddens, num_heads, dropout, bias=False, **kwargs):
    super().__init__()
    self.num_heads = num_heads
    self.save_hyperparameters()
    self.attention = utils.DotProductAttention(dropout=dropout)
    self.W_q = tf.keras.layers.Dense(units=num_hiddens, use_bias=bias)
    self.W_k = tf.keras.layers.Dense(units=num_hiddens, use_bias=bias)
    self.W_v = tf.keras.layers.Dense(units=num_hiddens, use_bias=bias)
    self.W_o = tf.keras.layers.Dense(units=num_hiddens, use_bias=bias)    

  def call(self, queries, keys, values, valid_lens, **kwargs):
    queries = self.W_q(queries) ## (batch_size, num_queries_per_sample, num_hiddens)
    keys = self.W_k(keys)       ## (batch_size, num_keys, num_hiddens)
    values = self.W_v(values)   ## (batch_size, num_values, num_hiddens)

    batch_size = queries.shape[0]
    last_dims = self.num_hiddens // self.num_heads

    ## Redo
    queries = self.transpose(queries)
    keys = self.transpose(keys)
    values =  self.transpose(values)

    if valid_lens is not None:
      valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)

    output = self.attention(queries, keys, values, valid_lens)
    ## (batch_size * num_heads, num_queries_per_sample, last_dims)

    output = self.reverse_transpose(output)
    return self.W_o(output)
  
  def transpose(self, X):
    ## X: (batch_size, num_queries/keys/values, num_hiddens)
    batch_size, num_qkv, num_hiddens = X.shape
    
    X = tf.reshape(X, shape=(batch_size, num_qkv, self.num_heads, -1))
    ## (batch_size, num_queries/keys/values, num_heads, num_hiddens//num_heads)
    
    X = tf.transpose(X, perm=[0, 2, 1, 3])
    ## (batch_size, num_heads, num_queries/keys/values, num_hiddens//num_heads)

    X = tf.reshape(X, shape=(-1, num_qkv, num_hiddens // self.num_heads))
    ## (batch_size * num_heads, num_queries/keys/values, num_hiddens//num_heads)

    return X
  
  def reverse_transpose(self, X):
    ## X: (batch_size*num_heads, num_queries/keys/values, num_hiddens//num_heads)
    batch_size_times_num_heads, num_qkv, num_hiddens_per_num_heads = X.shape
    
    batch_size = batch_size_times_num_heads // self.num_heads
    X  = tf.reshape(X, shape=(batch_size, self.num_heads, num_qkv, num_hiddens_per_num_heads))
    
    X = tf.transpose(X, perm=[0, 2, 1, 3])
    ## (batch_size, num_queries/keys/values, num_heads, num_hiddens//num_heads)

    X = tf.reshape(X, shape=(batch_size, num_qkv, self.num_hiddens))
    ## (batch_size, num_queries/keys/values, num_hiddens)

    return X
  
class PositionWiseFFN(tf.keras.layers.Layer):
  def __init__(self, ffn_num_hiddens, ffn_num_outputs):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(units=ffn_num_hiddens)
    self.relu = tf.keras.layers.ReLU()
    self.dense2 = tf.keras.layers.Dense(units=ffn_num_outputs)

  def call(self, X):
    return self.dense2(self.relu(self.dense1(X)))
  
class AddNorm(tf.keras.layers.Layer):
  """Residual Connection followed by layer normalization
     Normalization Shape: Shape of X excluding the first dimension (batch_size)
  """
  def __init__(self, norm_shape, dropout):
    super().__init__()
    self.dropout = tf.keras.layers.Dropout(dropout)
    self.ln = tf.keras.layers.LayerNormalization(axis=norm_shape)

  def call(self, X, Y, **kwargs):
    return self.ln(self.dropout(Y, **kwargs) + X)
  
class AttentionDecoder(utils.Decoder):
    """The base attention-based decoder interface.

    Defined in :numref:`sec_seq2seq_attention`"""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError