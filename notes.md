# Chapter 1: The Machine Learning Landscape

# Chapter 2: End-to-End Machine Learning Project

* California housing price analysis
	* Supervised
	* Regression
	* Batch

* Very cursory look at data
	* Check assumptions
	* Look for data limits
	* Plot histograms of all interval/ordinal variables to see rough distribution (skew and kurtosis)
	* Look at distribution of categorical variables
* Generate test set
	* Use a hash of a unique identifier for each observation so same test set can always be regenerated even after data has been filtered or appended
	* Can use random sampling (`sklearn.model_selection.train_test_split` is designed for this)
	* May want to consider stratified sampling (`sklearn.model_selection.StratifiedShuffleSplit`)

* Recode categorical variables as sets of binary variables

* 