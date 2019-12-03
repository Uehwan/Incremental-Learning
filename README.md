Hi


[test_newdrn.py]
1. Data format.
x is total input: [m][ch][input_dim]
sample is extracted from x: [ch][input_dim]
w is weight: [category][ch][input_dim]

2. Data processing.
(1) Synthetic data
-Read data
- Train, test.
   (Shuffled? it would be shuffled @ train phase, since train has shuffling.)

(2) UCI Real data
-Read file and save as cache
-shuffle the cache
-split cache to raw_data/class
-Format conversion from raw_data to data. 

So Data/raw_data/class is matched with order (shuffled from original order.)

*For batch learning, from r_study_list
   split 'raw_data' of list to train/test: 'batch_train' & 'batch_test'.
   For evaluation, answer category of 'batch_test' is saved as 'batch_test_class'.

3. Train / test
(1) KNN, GMM. 
Don't know about implementation inside.
Just gave cluster number as input
Data: batch_train/batch_test for each train/test phase .

(2) DRN, rDRN.
-Shuffling.
Specified to do shuffle @ train. (s_data, r_data all.)
-Data usage.
For training & testing in DRN/rDRN, data is used each once, total twice.
Reason for using twice despice online should be once is that KNN, GMM use twice so for fairness.
-online vs batch.
Data is inserted at once but is trained in online-manner inside the train/test.

4. If want to improve
-If change inserting really in online manner 
 Performance would be same but it would be more intuitive.
-Shuffle data_receiving phase / train twice?

5. Summary.
Issue could be
-online? data insertion manner in DRN/rDRN
-Shuffle twice?
-Data twice for online in DRN/rDRN?

