# DropOut用DeepLearningプログラム
import numpy as np
import scipy.io as sio
import cPickle
import gzip
import os
import sys
import time
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.printing
import theano.tensor.shared_randomstreams

from logistic_sgd import LogisticRegression
from load_data import load_umontreal_data, load_mnist, load_initial_params_data


##################################
## Various activation functions ##
##################################
#### rectified linear unit
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
#### sigmoid
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
#### tanh
def Tanh(x):
    y = T.tanh(x)
    return(y)
    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None,
                 use_bias=False):

        self.input = input
        self.activation = activation

        if W is None:
            W_values = np.asarray(rng.uniform(
                size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')
        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))
    
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, use_bias=use_bias)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class MLP(object):
    """A multilayer perceptron with all the trappings required to do dropout
    training.

    """
    def __init__(self,
            rng,
            input,
            layer_sizes,
            dropout_rates,
            activations,
            use_bias=True,
            initial_params=None,
            use_initial_params=False):

        #rectified_linear_activation = lambda x: T.maximum(0.0, x)

        # Set up all the hidden layers
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        next_layer_input = input
        #first_layer = True
        # dropout the input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        layer_counter = 0        
        for n_in, n_out in weight_matrix_sizes[:-1]:
            if use_initial_params :
                next_dropout_layer = DropoutHiddenLayer(rng=rng,
                        input=next_dropout_layer_input,
                        activation=activations[layer_counter],
                        W=initial_params[layer_counter],
                        n_in=n_in, n_out=n_out, use_bias=use_bias,
                        dropout_rate=dropout_rates[layer_counter + 1])
            else :
                next_dropout_layer = DropoutHiddenLayer(rng=rng,
                        input=next_dropout_layer_input,
                        activation=activations[layer_counter],
                        n_in=n_in, n_out=n_out, use_bias=use_bias,
                        dropout_rate=dropout_rates[layer_counter + 1])
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            # Reuse the paramters from the dropout layer here, in a different
            # path through the graph.
            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=activations[layer_counter],
                    # scale the weight matrix W with (1-p)
                    W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                    b=next_dropout_layer.b,
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            #first_layer = False
            layer_counter += 1
        
        # Set up the output layer
        n_in, n_out = weight_matrix_sizes[-1]
        if use_initial_params :
            dropout_output_layer = LogisticRegression(
                    input=next_dropout_layer_input,
                    W=initial_params[-1],
                    n_in=n_in, n_out=n_out)
        else :
            dropout_output_layer = LogisticRegression(
                    input=next_dropout_layer_input,
                    n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)

        # Again, reuse paramters in the dropout output.
        output_layer = LogisticRegression(
            input=next_layer_input,
            # scale the weight matrix W with (1-p)
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)
        self.layers.append(output_layer)

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors

        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors

        # Grab all the parameters together.
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]


def test_mlp(
        initial_learning_rate,
        learning_rate_decay,
        squared_filter_length_limit,
        n_epochs,
        batch_size,
        mom_params,
        activations,
        dropout,
        dropout_rates,
        results_file_name,
        layer_sizes,
        dataset,
        use_bias,
        initial_params,
        use_initial_params,
        random_seed=1234):
    """
    The dataset is the one from the mlp demo on deeplearning.net.  This training
    function is lifted from there almost exactly.

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


    """
    assert len(layer_sizes) - 1 == len(dropout_rates)
    
    # extract the params for momentum
    mom_start = mom_params["start"]
    mom_end = mom_params["end"]
    mom_epoch_interval = mom_params["interval"]
    
    
    datasets = load_umontreal_data(dataset)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]
    test_set_matrix_Y, train_set_matrix_Y = datasets[2]

    # compute number of minibatches for training, validation and testing
    number_of_train_set = train_set_x.get_value(borrow=True).shape[0]
    number_of_test_set = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches = number_of_train_set / batch_size
    n_test_batches = number_of_test_set / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    epoch = T.scalar()
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    y_matrix = T.matrix('y_matrix') # [int] labels

    learning_rate = theano.shared(np.asarray(initial_learning_rate,
        dtype=theano.config.floatX))

    rng = np.random.RandomState(random_seed)

    # construct the MLP class
    if use_initial_params :
        classifier = MLP(rng=rng, input=x,
                         layer_sizes=layer_sizes,
                         dropout_rates=dropout_rates,
                         activations=activations,
                         use_bias=use_bias,
                         initial_params=initial_params,
                         use_initial_params=use_initial_params)
    else :
        classifier = MLP(rng=rng, input=x,
                         layer_sizes=layer_sizes,
                         dropout_rates=dropout_rates,
                         activations=activations,
                         use_bias=use_bias)

    # Build the expresson for the cost function.
    #cost = classifier.negative_log_likelihood(y_matrix, batch_size)
    cost = classifier.negative_log_likelihood(y, batch_size)
    #dropout_cost = classifier.dropout_negative_log_likelihood(y_matrix, batch_size)
    dropout_cost = classifier.dropout_negative_log_likelihood(y, batch_size)

    # Compile theano function for testing.
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})

    train_error_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    #theano.printing.pydotprint(test_model, outfile="test_file.png",
    #        var_with_name_simple=True)

    #theano.printing.pydotprint(validate_model, outfile="validate_file.png",
    #        var_with_name_simple=True)

    # Compute gradients of the model wrt parameters
    gparams = []
    for param in classifier.params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(dropout_cost if dropout else cost, param)
        gparams.append(gparam)

    # ... and allocate mmeory for momentum'd versions of the gradient
    gparams_mom = []
    for param in classifier.params:
        gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)

    # Update the step direction using momentum
    updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(classifier.params, gparams)
    ]

    # Compile theano function for training.  This returns the training cost and
    # updates the model parameters.
    output = dropout_cost if dropout else cost
    train_model = theano.function(inputs=[index], outputs=output,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                #y_matrix: train_set_matrix_Y[index * batch_size:(index + 1) * batch_size]})
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(train_model, outfile="train_file.png",
    #        var_with_name_simple=True)

    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    #decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
    #        updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    best_params = None
    best_test_errors = np.inf
    best_iter = 0
    epoch_counter = 0
    start_time = time.time()
    error_matrix = np.zeros([n_epochs, 4])

    #results_file = open(results_file_name, 'wb')

    while epoch_counter < n_epochs:
        # Train this epoch
        epoch_counter = epoch_counter + 1

        if (epoch_counter % 10000) == 0 :
            params = [classifier.params[0].get_value(), classifier.params[1].get_value(), classifier.params[2].get_value()]
            sio.savemat('initial_params', {'epochs':epoch_counter, 'params':params})

        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
        # Compute loss on validation set
        test_losses = [test_model(i) for i in xrange(n_test_batches)]
        train_losses = [train_error_model(i) for i in xrange(n_train_batches)]
        test_losses_sum = np.sum(test_losses)
        train_losses_sum = np.sum(train_losses)
        this_test_errors = test_losses_sum / float(number_of_test_set)
        this_train_errors = train_losses_sum / float(number_of_train_set)

        # Report and save progress.
        print "epoch {}, test error {}, train error {}, train cost {}, learning_rate={}{}".format(
                epoch_counter, this_test_errors, this_train_errors, minibatch_avg_cost,
                learning_rate.get_value(borrow=True),
                " **" if this_test_errors < best_test_errors else "")
        error_matrix[epoch_counter - 1] = epoch_counter, this_train_errors, this_test_errors, minibatch_avg_cost 

        if this_test_errors < best_test_errors :
            best_test_errors = this_test_errors
            best_iter = epoch_counter

        #results_file.write("{0}\n".format(this_test_errors))
        #results_file.flush()

        #new_learning_rate = decay_learning_rate()

    end_time = time.time()
    params = [classifier.params[0].get_value(), classifier.params[1].get_value(), classifier.params[2].get_value()]
    print(('Optimization complete. Best test score of %f %% '
           'obtained at iteration %i') %
          (best_test_errors * 100., best_iter))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    sio.savemat(results_file_name, {'learning_rate':initial_learning_rate, 'layer_sizes':layer_sizes, 'dropout_rates':dropout_rates, 'batch_size':batch_size, 'bias':use_bias, 'epochs':n_epochs, 'error_matrix':error_matrix, 'best_test_errors':best_test_errors * 100., 'best_iter':best_iter, 'elapsed_time':(end_time - start_time) / 60., 'params':params})


if __name__ == '__main__':
    import sys
    
    # set the random seed to enable reproduciable results
    # It is used for initializing the weight matrices
    # and generating the dropout masks for each mini-batch
    random_seed = 1234

    initial_learning_rate = 0.5
    learning_rate_decay = 0.998
    squared_filter_length_limit = 15.0
    n_epochs = 50000
    batch_size = 100
    layer_sizes = [ 647, 500, 30 ]
    dropout_hidden_rate = np.float64(sys.argv[3])
    
    # dropout rate for each layer
    dropout_rates = [ 0, dropout_hidden_rate ]
    # activation functions for each layer
    # For this demo, we don't need to set the activation functions for the 
    # on top layer, since it is always 10-way Softmax
    activations = [ Tanh ]
    
    #### the params for momentum
    mom_start = 0.000001
    mom_end = 0.00001
    # for epoch in [0, mom_epoch_interval], the momentum increases linearly
    # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
    mom_epoch_interval = 500
    mom_params = {"start": mom_start,
                  "end": mom_end,
                  "interval": mom_epoch_interval}
                  
    dataset = 'VisionHogFeatures.mat'
    params_dataset = 'initial_params0_1.mat'
    initial_params = load_initial_params_data(params_dataset);
    results_file_name = sys.argv[2]

    if len(sys.argv) < 4:
        print "Usage: {0} [dropout|backprop]".format(sys.argv[0])
        exit(1)

    elif sys.argv[1] == "dropout":
        dropout = True
        #results_file_name = "results_dropout.txt"

    elif sys.argv[1] == "backprop":
        dropout = False
        #results_file_name = "results_backprop.txt"

    else:
        print "I don't know how to '{0}'".format(sys.argv[1])
        exit(1)

    test_mlp(initial_learning_rate=initial_learning_rate,
             learning_rate_decay=learning_rate_decay,
             squared_filter_length_limit=squared_filter_length_limit,
             n_epochs=n_epochs,
             batch_size=batch_size,
             layer_sizes=layer_sizes,
             mom_params=mom_params,
             activations=activations,
             dropout=dropout,
             dropout_rates=dropout_rates,
             dataset=dataset,
             results_file_name=results_file_name,
             use_bias=False,
             initial_params=initial_params,
             use_initial_params=True,
             random_seed=random_seed)

