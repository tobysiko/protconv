from keras import backend as K
from keras.layers import Layer
from keras.activations import softmax

class Conv_Seq_Dec_Softmax_Layer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Conv_Seq_Dec_Softmax_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Conv_Seq_Dec_Softmax_Layer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # assume shape (None, 1600, 22)
        #....?
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)