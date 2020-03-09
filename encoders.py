import tensorflow as tf
import tensorflow.keras.layers as L


class WordRNN(tf.keras.layers.Layer):
    def __init__(self,vocab_size,emb_size,hid_size,dropout=0.1,recurrent_dropout=0.2):
        super(WordRNN,self).__init__()

        self.embed = L.Embedding(vocab_size,emb_size)
        self.gru = L.GRU(hid_size,dropout=dropout,recurrent_dropout=recurrent_dropout)
        self.bi_gru = L.Bidirectional(self.gru)
        self.fc = L.Dense(1)

    def __call__(self,X,test=False):
        if not test:
            return self.bi_gru(self.embed(X))
        else:
            return self.fc(self.bi_gru(self.embed(X)))

    def get_logits(self,batch_lines,transformer):
        '''
        input a batch of text, return logits
        '''
        batch_ixs = transformer.to_matrix(batch_lines)
        return self(batch_ixs,test=True)

class CharRNN(tf.keras.layers.Layer):
    '''
    concatenate outputs of different CNN layers to capture information of n-grams with different ns.
    conv_kernel_sizes is a list of convolutional kernel sizes.
    '''
    def __init__(self,ch_size,emb_size,hid_size,conv_hid_size,conv_kernel_sizes,
                dropout=0.1,recurrent_dropout=0.2):
        super(CharRNN,self).__init__()

        self.embed = L.Embedding(ch_size,emb_size)
        convs = []
        for size in conv_kernel_sizes:
            convs.append(L.Conv1D(conv_hid_size,size,padding='same'))
        self.convs = convs
        self.gru = L.GRU(hid_size,dropout=dropout,recurrent_dropout=recurrent_dropout)
        self.fc = L.Dense(1)

    def __call__(self,X,test=False):
        X_embed = self.embed(X)
        X_convs = []
        for conv_layer in self.convs:
            X_convs.append(conv_layer(X_embed))
        X_input = L.Concatenate()(X_convs)

        if not test:
            return self.gru(X_input)
        else:
            return self.fc(self.gru(X_input))

    def get_logits(self,batch_lines,transformer):
        batch_ixs = transformer.to_matrix(batch_lines)
        return self(batch_ixs,test=True)


#=================================================
#combine word-level and char-level embeddings using Highway Network
#=================================================
class HighwayNetwork(tf.keras.layers.Layer):
    '''
    z = t * tanh(WH @ X + bH) + (1 - t) * X
    t = sigmoid(WT @ X + bT)
    '''
    def __init__(self,hid_size):
        super(HighwayNetwork,self).__init__()
        #WH, WT are square matrices
        self.dense_H = L.Dense(hid_size,activation='tanh')
        self.dense_T = L.Dense(hid_size,activation='sigmoid')
    
    def __call__(self,X):
        #t has the same shape as X
        t = self.dense_T(X)
        return t * self.dense_H(X) + (1 - t) * X


class CombinedEncoder(tf.keras.Model):
    def __init__(self,vocab_size,char_size,emb_size_w,emb_size_c,hid_size_w,hid_size_c,
                conv_kernel_sizes_c,conv_hid_size_c,dropout=0.1,recurrent_dropout=0.2):
        super(CombinedEncoder,self).__init__()

        self.w_encoder = WordRNN(vocab_size,emb_size_w,hid_size_w,
                                 dropout,recurrent_dropout)
        
        self.c_encoder = CharRNN(char_size,emb_size_c,hid_size_c,conv_hid_size_c,
                                 conv_kernel_sizes_c,dropout,recurrent_dropout)

        #used for binary classification
        #project sentence vectors to probabilities, here dimension=1
        self.fc = L.Dense(1)
        
    def __call__(self,X_w,X_c):
        X_word = self.w_encoder(X_w)
        X_char = self.c_encoder(X_c)

        X_input = L.Concatenate()([X_word,X_char])

        highway = HighwayNetwork(X_input.shape[1])

        logits = self.fc(highway(X_input))

        return logits
    
    def get_logits(self,batch_lines,transformer_word,transformer_char):
        batch_ixs_word = transformer_word.to_matrix(batch_lines)
        batch_ixs_char = transformer_char.to_matrix(batch_lines)
        return self(batch_ixs_word,batch_ixs_char)


#=================================================
#word-level embeddings + 3 convolution layers with different kernel sizes + GRU + Attention
#=================================================
class Attention(tf.keras.layers.Layer):
    def __init__(self,attention_hid_size):
        super(Attention,self).__init__()
        self.attention_hid_size = attention_hid_size
        self.W1 = L.Dense(self.attention_hid_size)
        self.W2 = L.Dense(self.attention_hid_size)
        self.V = L.Dense(1)
        
    def __call__(self,query,values):
        '''
        query: hidden state
        values: encoder's output at each timestep
        peek at encoder's all outpus to globally use input information
        '''
        query = tf.expand_dims(query,1)
        
        score = self.V(tf.nn.tanh(self.W1(query)+self.W2(values)))
        ###ATTENTION, softmax on time step axis
        attention_weights = tf.nn.softmax(score,axis=1)
        
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector,axis=1)
        
        return context_vector, attention_weights


class Encoder(tf.keras.Model):
    def __init__(self,vocab_size,emb_size,hid_size,attention_hid_size):
        super(Encoder,self).__init__()
        
        self.hid_size = hid_size
        self.embed = L.Embedding(vocab_size,emb_size)
        self.conv_1 = L.Conv1D(128,3,padding='same')
        self.conv_2 = L.Conv1D(128,5,padding='same')
        self.conv_3 = L.Conv1D(128,7,padding='same')
        self.gru = L.GRU(hid_size,return_sequences=True,
                                       return_state=True,recurrent_initializer='glorot_uniform',
                                       dropout=0.2,recurrent_dropout=0.1)
        self.attention = Attention(attention_hid_size)
        self.fc_1 = L.Dense(32,activation='elu')
        self.dropout = L.Dropout(rate=1)
        self.fc_2 = L.Dense(1)
        
    def __call__(self,X,initial_hidden_state=None):
        if initial_hidden_state is None:
            bs = X.shape[0]
            initial_hidden_state = self._init_hidden_state(bs)
        
        inputs = self.embed(X)
        conv_outputs_1, conv_outputs_2, conv_outputs_3 = self.conv_1(inputs), self.conv_2(inputs), self.conv_3(inputs)
        enc_input = L.Concatenate()([conv_outputs_1,conv_outputs_2,conv_outputs_3])
        
        enc_outputs, state = self.gru(enc_input,initial_state=initial_hidden_state)
        context_vector, _ = self.attention(state,enc_outputs)
        
        logits = self.fc_2(self.dropout(self.fc_1(context_vector)))
        
        return logits
        
    def _init_hidden_state(self,batch_size):
        return tf.zeros((batch_size,self.hid_size))

    def get_logits(self,batch_lines,transformer):
        batch_ixs = transformer.to_matrix(batch_lines)
        return self(batch_ixs)