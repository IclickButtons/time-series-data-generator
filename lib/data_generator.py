import numpy as np 
import pandas as pd 
import operator 

class DataGeneratorTimeSeries(object): 
    '''
    Args: 
        
    '''
    def __init__(self, num_time_steps, num_pred, batch_size, data_dir,   
                 output_shape = 'BSD', shuffle=False, last_batch_dif=False): 
        
        self.data_dir = data_dir 
        self.num_time_steps = num_time_steps 
        self._num_pred = num_pred
        self.batch_size = batch_size
        self._shuffle = shuffle 
        self._last_batch_dif = last_batch_dif
        self._data = self._load_data()  
        self._data_dim = self._data.shape[1]
        self._data_length = len(self._data) 
	
	# number of time windows that can be generated from the 
	# data 
        self._segments = (self._data_length - self._num_time_steps 
                          - (self._num_pred+1))          

        self._cursor_list = list(range(0, self._segments + 2)) 
        self._cursor_pos = None 

        self.output_shape = output_shape 
    
    data_dir = property(operator.attrgetter('_data_dir')) 

    @data_dir.setter 
    def data_dir(self, dd): 
        if not dd: 
            raise Exception('no data directory was specified') 
        else: 
            self._data_dir = dd
    
    
    output_shape = property(operator.attrgetter('_output_shape')) 

    @output_shape.setter 
    def output_shape(self, os): 
        if os == 'BSD': 
            self._output_shape = 1 
        elif os == 'SBD': 
            self._output_shape = 2 
        else: 
            raise Exception('unknown keyword argument for output_shape') 
    
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

    def _load_data(self): 
        #data = np.genfromtxt(self._data_dir, delimiter=',', skip_header=1)     
        data = pd.read_csv(self._data_dir)
        data = data.values 
        return data 
    
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
    
    def convert_output_format(self, batch): 
        array = np.asarray(batch) 
        if self._output_shape == 1: 
            array = array.reshape(-1, self._num_time_steps, self._data_dim) 
        elif self._output_shape == 2: 
            array = array.reshape(self._num_time_steps, -1, self._data_dim) 

        return array 

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
        batch_x = self.convert_output_format(batch_x)  
        batch_y = np.asarray(batch_y) 
        batch_y = batch_y.reshape(-1, 1)

        return {'x_train': batch_x, 'y_train':  batch_y} 
    

    def reset_gen(self): 
        '''resets the generator by transforming the cursor list to its 
        original state
        '''

        # reset variables to orginal values
        self._cursor_list = list(range(0, self._segments + 2)) 
        self._cursor_pos = None 

    def yield_batches(self): 
        if self._last_batch_dif==False: 
            if (len(self._cursor_list) >= self._batch_size): 
                return True 
        if self._last_batch_dif==True: 
            if (len(self._cursor_list) >= 0): 
                return True 
