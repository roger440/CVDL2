Team members : Vilceanu Ovidiu, ALexandru Mocanu

Components:
	1. Setup
			This is the component we used in order to prepare the data for the BERT model to work with.
		The data is filtered, excluding the records with invalid format, and grouped into folders used
		for training and testing, each categorised in folders for classification
	2. Main
		This is the part in which we load the data using a tensorflow data loader, download the 'Small Bert'
		model, configure the loss function, the optimiser, hper-parameters, and use the training set in order to
		fine-tune the already pre-trained BERT model. The validation set will be used in order to compute the loss for
		the current epoch. For each epoch, details regarding the loss value and the accuracy are printed

BERT:
	Bert is a machine learning framework used for solving natural language processing problems. It achieves this in 2 steps:
	-pre-training: a phase in which BERT uses unsupervised learning in order to understand language and context
	-fine-tuning: slightly modifying the parameters obtained in pre-training, uses supervised learning to corelate input to output