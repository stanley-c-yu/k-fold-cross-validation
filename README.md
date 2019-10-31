# k-fold-cross-validation
Implementation of K-Fold Cross-Validation sans SciKit Learn's implementation of KFCV testing with SciKit Learn's implementation of Linear Discriminant Analysis and Logistic Regression.  

Makes use of the Spambase Dataset courtesy of UC Irvine's Machine Learning Repository.  

The Spambase Dataset consists of data points regarding over 5000 emails classified as either SPAM ("junk") or HAM.  It has 57 columns corresponding to feature data, and a 58th column consisting of a binary response, where 1 denotes SPAM and 0 denotes HAM.  

Data is contained in the enclosed "spambase.names" file, which is unlabeled.  The column names, in order, are included in the "spambase.names" file.  More complete information regarding the dataset is included in the "spambase.documentation" file.  

The "kfcv.py" file first reads in and pre-processes the data.  It then creates instances of the Logistict Regression and Linear Discriminant Analysis models, which it employs in the following KFCV function for testing purposes.  
