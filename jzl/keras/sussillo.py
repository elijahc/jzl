from keras.engine.topology import Layer
from keras.engine.topology import InputSpec
from keras import activations
from keras import initializers
from keras import backend as K
from keras.layers.recurrent import Recurrent
import numpy as np

class Sussillo(Recurrent):

    def __init__(self, units,
                 activation='tanh',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='glorot_uniform',

                 **kwargs):

        super(Force, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.state_spec = InputSpec(shape=(None, self.units))

    def build(self, input_shape):
        # MUST define self.input_spec and self.state_spec with
        # complete input shapes

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec[0] = InputSpec(shape=(batch_size,None,self.input_dim))
        self.states = [None]

        if self.stateful:
            self.reset_states()

        self.kernel = self.add_weight(
                shape=(self.input_dim, self.units),
                name='kernel',
                initializer=self.kernel_initializer)

        self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                name='recurrent_kernel',
                initializer=self.recurrent_initializer)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        name='bias',
                                        initializer=self.bias_initializer,)
        else:
            self.bias = None

        self.built = True

    def step(self, inputs, states, dt=0.1):
        h1 = states[0] # Previous State
        r1 = self.activation(h1) # Previous firing rates

        x_h = K.dot(inputs, self.kernel)
        r_h = K.dot(r1,self.recurrent_kernel)

        h = (1.0-dt)*h1 + r_h*dt + x_h*dt
        if self.use_bias:
            h = K.bias_add(h, self.bias)

        r = self.activation(h)

        return r, [h]


    def compute_output_shape(self, input_shape):
        if isinstance(input_shape,list):
            input_shape = input_shape[0]

        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], self.units)
        else:
            output_shape = (input_shape[0],self.units)

        if self.return_state:
            state_shape = [(input_shape[0],self.units) for _ in self.states]
            return [output_shape] + state_shape
        else:
            return output_shape

