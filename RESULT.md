# Final Results

Final Model Results:

First Header  | Second Header
------------- | -------------
Overall Accuracy  | 0.538
Micro Precision   | 0.538
Macro Precision   | 0.550
Micro Recall      | 0.538
Macro Recall      | 0.633

***Accuracy*** refers to the fraction of preditions the model got correct on the test dataset. It is the sum of true positives and true negatives divided by all results.
***Precision*** refers to the model's ability to correctly identify a positive, it is the fraction of true positives and all positives. 
***Recall*** refers to the model's ability to correctly class the data. It is the fraction of true positives over the sum of true positive and false negatives. 

# Reflection 

As seen in the table above, all metrics are above 50% so while the model is not very accurate it performs much better than a random guess which has a 33% chance of being correct given there are three categories (easy, medium, and challenging). The model's recall is about 10% higher than the precision meaning that about 63% of the recipes are correctly captured into the category it belongs to, i.e. if a recipe is medium difficulty there is a 63% chance it will be put into the medium difficulty class. However, the lower precision indicates that the model has a 45% chance of incorrectly identifying that a recipe belongs in a certain category when it does not. 

Through some experimentation it was found that the following factors could impact this model's accuracy, precision, and recall:
* Size of the dataset
* Fraction of the dataset used for training
* Minimum document frequency when extracting n-grams (the minimum number of documents that an n-gram has to appear in to be included in the document term matrix created when extracting n-gram features from text)

Initially a dataset containing 30 recipes was used. At this low value the precision and recall were about 40% as there was not enough training data to provide meaningful information to the model. Additionally, the minimum document frequency had to be decreased to about 2 to 3 because there were only about 10 recipes per class and increasing the minimum document frequency would have been too limiting. However, by having such a low minimum document frequency I was determining that many features that might not have been significant were significant enough for consideration by the model. Thus, the data was potentially messy. 
Conversely, a larger dataset with 100 recipes also resulted in lower overall accuracy. This may have been due to over training creating bias in the model. 



