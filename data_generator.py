import numpy as np 
import operator 

class DataGeneratorTimeSeries(object): 
    '''TODO:implement output format, implement multidimensionality, last batch
    different length'''
    def __init__(self, data, num_time_steps, num_pred, batch_size, 
                 shuffle=False, variable_batch_size=False): 
        
        self._data = data 
        self.num_time_steps = num_time_steps 
        self._num_pred = num_pred
        self.batch_size = batch_size
        self._shuffle = shuffle 
        self._variable_batch_size = variable_batch_size 

        self._data_dim = self._data.shape[1]
        self._data_length = len(self._data) 
	
	# number of time windows that can be generated from the 
	# data 
        self._segments = (self._data_length - self._num_time_steps 
                          - (self._num_pred+1))          

        self._cursor_list = list(range(0, self._segments + 2)) 
        self._cursor_pos = None 

    batch_size = property(operator.attrgetter('_batch_size')) 
        
    @batch_size.setter 
    def batch_size(self, bs):
        if not (bs > 0): 
            raise Exception('batch size has to be greater than zero') 
        self._batch_size = bs 

    num_time_steps = property(operator.attrgetter('_num_time_steps')) 

    @num_time_steps.setter 
    def num_time_steps(self, nt): 
        if not (nt > 0): 
            raise Exception('sequence length has to be greater than zero')
        self._num_time_steps = nt 

    num_pred = property(operator.attrgetter('_num_pred')) 

    @num_pred.setter 
    def num_pred(self, np): 
        if not (np > 0): 
            raise Exception('length of prediction sequence has to greater than zero')
        
    def _next_sequence(self): 
        '''Creates training sequences (x) of length num_time_steps and
        groundtruth predictions (y) of length num_pred. In this 
        implementation the cursor list represents all possible start positions of 
        sequences. After a sequence is created its start position is 
        removed from the cursor list. 
        '''

        # the keyword shuffle determines if the sequence starts at 
        # a random postion in the data otherwise the first poistion in the
        # cursor list is used 
        if self._shuffle: 
            self._cursor_pos = random.choice(self._cursor_list) 
        else: 
            self._cursor_pos = self._cursor_list[0]

        # create numpy sequence placeholders
        seq_x = np.zeros((self._num_time_steps, self._data_dim), dtype=np.float32) 
        seq_y = np.zeros((self._num_pred), dtype=np.float32) 
        
        # create x sequence
        for i in range(self._num_time_steps): 
            seq_x[i] = self._data[self._cursor_pos + i,:]
        
        # create y seqeuence 
        seq_y[0] = self._data[self._cursor_pos + self._num_time_steps, 0] 
       
        # remove cursor position from cursor list 
        self._cursor_list.remove(self._cursor_pos) 

        return seq_x, seq_y 

    def create_batches(self): 
        '''creates batches of training sequences and groundtruth predictions of
        specified sizes
        '''

        batch_x = []
        batch_y = [] 
        
        # if (last) batch can not be filled with enough sequences only 
        # append the remaining sequences  
        if (len(self._cursor_list) < self._batch_size): 
            for i in range(len(self._cursor_list)): 
                x, y = self._next_sequence() 
                batch_x.append(x)
                batch_y.append(y) 

        else: 
            for i in range(self._batch_size): 
                x, y = self._next_sequence()
                batch_x.append(x) 
                batch_y.append(y) 

        # convert batches to correct output format 
        batch_x = np.asarray(batch_x) 
        batch_x = batch_x.reshape(-1, self._num_time_steps, self._data_dim) 
        batch_y = np.asarray(batch_y) 
        batch_y = batch_y.reshape(-1, 1)

        return batch_x, batch_y 

    def reset_gen(self): 
        '''resets the generator by transforming the cursor list to its 
        original state
        '''

        # reset variables to orginal values
        self._cursor_list = list(range(0, self._segments + 2)) 
        self._cursor_pos = None 

    def yield_batches(self): 
        if (len(self._cursor_list) >= self._batch_size): 
            return True 