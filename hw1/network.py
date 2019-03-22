import tensorflow as tf
import numpy as np 

class Network(object):
    def __init__(self, input_data, output_data, batch_size = 512):
        