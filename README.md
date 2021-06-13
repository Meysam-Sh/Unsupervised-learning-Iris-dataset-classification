This is an impementation of Expectation Maximization algorithm to cluster Iris dataset https://en.wikipedia.org/wiki/File:Iris_dataset_scatterplot.svg. The Iris dataset is a well-known example, with three types of Irises which can be classified using 4-dimensional measurements (sepal length, sepal width, petal length, petal width).

The Iris dataset can at least be partially clustered using even just one dimension. Here, I at first I use the Petal Length feature and apply the 1D version of the EM algorithm to cluster the dataset to two or three groups. Then, I use all dataset with all features to cluster the dataset to three groups: 

**Two clusters for one dimension (Petal Length)**
 Considering the scatterplots of iris dataset, we can classify the data to two clusters regarding "Petal Length" dimension. In this case, Versicolor and Virginia flowers are get grouped together.

We properly initialized the mean and standard deviation of each cluster. Then implemented EM algorithm. We also calculated log likelihood. Based on the convergence of the log-likelihood we selected an appropriate number of iterations to run. 

**Three clusters for all dimensions (whole of dataset)**
In this case, we defined three Gaussian distributions which are initialized with proper means and standard deviations. In the implementation, the dataset is clustered based on each dimension iteratively.

**Three cluster for one dimension (Petal Width)**
Here, three Gaussian distributions are considered to classify dataset based one the fourth dimension which is "Petal Width". According to dataset scatterplot, it can be suggested that Versicolor and Virginia flowers are still a bit hard to be classified. 
