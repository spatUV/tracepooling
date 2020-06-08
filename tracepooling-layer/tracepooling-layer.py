import tensorflow as tf

class TraceLayer(Layer):

    def __init__(self, downsample, **kwargs): 
        super(TraceLayer, self).__init__(**kwargs)
        self.downsample = downsample
    
    def build(self, input_shape):
        
        if len(input_shape) == 4:
            self.filt = input_shape[2]*input_shape[3]
            self.freq = input_shape[2]
            self.channel = input_shape[3]
        else:
            self.filt = input_shape[2]
        

        super(TraceLayer, self).build(input_shape)  # Be sure to call this at the end

    # Where the forward-pass operation is implemented
    # Logic of the layer
    def call(self, x):
        #because when init batch value is unknown 'none'
        shape_before_pool = tf.shape(x)
        if shape_before_pool.shape[0] > 3:
            self.batches = tf.shape(x)[0]
            self.frames = tf.shape(x)[1]
            x = tf.reshape(x, [self.batches, self.frames, int(self.filt)])
            output = self.get_trace(x)
            output = tf.reshape(output,[self.batches, -1, self.freq, self.channel])
        else:
            self.batches = tf.shape(x)[0]
            self.frames = tf.shape(x)[1]
            output = self.get_trace(x)
            
        return output

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 4:
            output_shape = [(input_shape[0], input_shape[1], input_shape[2], input_shape[3])]
        else:
            output_shape = [(input_shape[0], input_shape[1], input_shape[2])]
        
        return output_shape

    def get_config(self):
        config = super(TraceLayer, self).get_config()
        config['downsample'] = self.downsample
               
        return config
    
    def get_trace(self, feat):
        npoints = tf.cast(tf.floor(self.frames/self.downsample),tf.int32)
               
        aux1 = tf.slice(feat,[0, 0, 0], [self.batches, self.frames-1, self.filt])
        aux2 = tf.slice(feat,[0, 1, 0], [self.batches, self.frames-1, self.filt])
        
        #Calculate the energy of the filters in order to get the trace
        aux1E = tf.reduce_sum(aux1*aux1, axis=2)
        aux2E = tf.reduce_sum(aux2*aux2, axis=2)     
        dif  = tf.subtract(aux2E,aux1E)
        
        dif_conc = tf.concat([tf.zeros((self.batches,1),dtype='float32'),dif],1)        
        LT =tf.cast(tf.cumsum(tf.cast(tf.sqrt(dif_conc * dif_conc),tf.float64),axis=1),tf.float32)
        
        LT_norm = tf.divide(LT,tf.expand_dims(LT[:,-1],1))

        # Calculate new index points based on a suboptimal approximation
        LT_aux1 = tf.slice(LT_norm,[0, 0], [self.batches, self.frames-1])
        LT_aux2 = tf.slice(LT_norm,[0, 1], [self.batches, self.frames-1])
        LT_dif  = tf.subtract(LT_aux2, LT_aux1)
        
        # Calculate new indices
        vals, indices = tf.nn.top_k(LT_dif,npoints-1)
        whichs = tf.reduce_sum(tf.one_hot(indices,self.frames-1),axis=1)
        
        index_points = tf.cast(tf.cumsum( tf.cast(tf.concat( [tf.zeros([self.batches,1],dtype='float32') , whichs] ,1),tf.float64), axis=1),tf.int32)

        # Flatten the indices to use them as segment section indicators               
        values = tf.reshape(tf.range(self.filt*self.batches),[-1,1])
        S = tf.reshape(tf.tile(npoints*values,[1,self.frames]),[-1])
        I = tf.reshape(tf.tile(index_points,[1, self.filt]),[-1])

        M = tf.reshape(tf.transpose(feat,[0,2,1]),[-1])
        Add = I+S
        C = tf.math.segment_max(M, Add)
        output = tf.transpose(tf.reshape(C,[self.batches,self.filt,npoints]), [0,2,1])
        

        return output