from unittest import TestCase
from data_generator import DataGeneratorTimeSeries
import numpy as np 
import pandas as pd 

class DataGeneratorTimeSeriesTest(TestCase): 
    
    def setUp(self):
        # create sample data frame and values 
        dim = 1
        data_array = np.array([np.arange(0,50) for i in
            range(dim)]).reshape(50,dim) 
        self._test_df = pd.DataFrame(data_array, columns=list('A'))
        self._test_data = self._test_df.values 
        self._generator = DataGeneratorTimeSeries(self._test_data, 5, 1, 15,
                                                  output_shape = 'BSD')  
    
    def test_correct_y_batches(self): 
        i = 0 
        while self._generator.yield_batches(): 
            x, y = self._generator.create_batches()
            x_true = np.array([[self._generator._num_time_steps + j + i] 
                               for j in range(self._generator._batch_size)])
            x_true = x_true.tolist() 

            self.assertEqual(y.tolist(), x_true, 
                             'incorrect generated y values') 
            i += self._generator._batch_size
    
    def test_correct_x_batches(self): 
        k = 0 
        while self._generator.yield_batches(): 
            x, y = self._generator.create_batches() 
            x_true = []
            
            for j in range(self._generator._batch_size): 
                x_seq = [[i] for i in self._test_data[j + k:
                         self._generator._num_time_steps + j + k]] 
                x_true.append(x_seq) 

            self.assertEqual(x.tolist(), x_true, 
                             'incorrect generated x values') 
            k += self._generator._batch_size

    def test_reset_generator(self): 
        while self._generator.yield_batches(): 
            x, y = self._generator.create_batches()
        self._generator.reset_gen()
        original_cursor_list = list(range(0, self._generator._segments 
                                           + 2)) 
        self.assertEqual(self._generator._cursor_list, original_cursor_list, 
                         'generator did not reset correctly')  
