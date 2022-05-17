"""
Created on Fri Jun 26 17:27:26 2015

@author: Gabriel Moreira
Adapted from https://github.com/hidasib/GRU4Rec
"""

import time
import numpy as np
import pandas as pd


def evaluate_sessions_batch_neg_samples(pr, streaming_metrics, test_data, items=None, cut_off=20, batch_size=100,
    session_key='SessionId', item_key='ItemId', time_key='Time', session_neg_samples_key='neg_samples'):
    '''
    Evaluates the GRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.

    Parameters
    --------
    pr : gru4rec.GRU4Rec
        A trained instance of the GRU4Rec network.
    streaming_metrics : list
        A list of metric classes providing the proper methods
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation. If it is set high, the memory consumption increases. Default value is 100.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')    
    session_neg_samples_key: string
        Header of the list column with the negative samples for the session (Sampled during NAR module training) (default: 'neg_samples')

    Returns
    --------
    out : list of tuples
        (metric_name, value)
    
    '''

    #IMPORTANT: Filtering for prediction only items present in train set, to avoid errors on GRU4REC
    #test_data = pd.merge(test_data, pd.DataFrame({'ItemIdx':pr.itemidmap.values, item_key:pr.itemidmap.index}), on=item_key, how='inner')
    #test_data = test_data.groupby(session_key).filter(lambda x: len(x) > 1)

    actions = len(test_data)
    sessions = len(test_data[session_key].unique())
    print('START batch eval ', actions, ' actions in ', sessions, ' sessions') 
    sc = time.clock();
    st = time.time();

    for m in streaming_metrics:
        m.reset()

    pr.predict = None #In case someone would try to run with both items=None and not None on the same model without realizing that the predict function needs to be replaced
    test_data.sort_values([session_key, time_key, item_key], inplace=True)
    offset_sessions = np.zeros(test_data[session_key].nunique()+1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()

    if len(offset_sessions) - 1 < batch_size:
        batch_size = len(offset_sessions) - 1
        
    iters = np.arange(batch_size).astype(np.int32) 
    
    maxiter = iters.max()    
    start = offset_sessions[iters]
    end = offset_sessions[iters+1]
    
    in_idx = np.zeros(batch_size, dtype=np.int32)    
    np.random.seed(42)
    
    while True:
        
        valid_mask = iters >= 0
        if valid_mask.sum() == 0:
            break
        
        start_valid = start[valid_mask]    

        minlen = (end[valid_mask]-start_valid).min()
        in_idx[valid_mask] = test_data[item_key].values[start_valid]

        for i in range(minlen-1):
  

            out_idx = test_data[item_key].values[start_valid+i+1]
            neg_samples = test_data[session_neg_samples_key].values[start_valid+i+1]
            
            if items is not None:
                uniq_out = np.unique(np.array(out_idx, dtype=np.int32))
                preds = pr.predict_next_batch(iters, in_idx, np.hstack([items, uniq_out[~np.in1d(uniq_out,items)]]), batch_size)
            else:
                preds = pr.predict_next_batch(iters, in_idx, None, batch_size)
            
                
            preds.fillna(0, inplace=True)
            in_idx[valid_mask] = out_idx  
            
            j=0
            for part, series in preds.loc[:,valid_mask].iteritems(): 
                
                #Combining the next clicked item with a list of negative samples
                items_to_predict = [out_idx[j]] + neg_samples[j]
                
                preds_items = preds.loc[items_to_predict].sort_values( part, ascending=False)[part]                                

                preds_for_metrics = np.expand_dims(np.expand_dims(preds_items.index.values, 0), 0)
                label_for_metrics = np.array([[out_idx[j]]])                
                for m in streaming_metrics:
                    m.add(preds_for_metrics, label_for_metrics)                    

                j += 1
            
        start = start+minlen-1
        mask = np.arange(len(iters))[(valid_mask) & (end-start<=1)]
        for idx in mask:
            maxiter += 1
            if maxiter >= len(offset_sessions)-1:
                iters[idx] = -1
            else:
                iters[idx] = maxiter
                start[idx] = offset_sessions[maxiter]
                end[idx] = offset_sessions[maxiter+1]
                
    print( 'END batch eval ', (time.clock()-sc), 'c / ', (time.time()-st), 's' )
    
    metric_results = {}
    for m in streaming_metrics:
        metric_results[m.name()] = m.result()
    
    return metric_results