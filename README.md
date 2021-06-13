This is an implementation of the Expectation-Maximization algorithm to cluster Iris dataset https://en.wikipedia.org/wiki/File:Iris_dataset_scatterplot.svg. The Iris dataset is a well-known example, with three types of Irises which can be classified using 4-dimensional measurements (sepal length, sepal width, petal length, petal width).

The Iris dataset can be partially clustered using even just one dimension. Here, at first, I use the Petal Length feature and apply the 1D version of the EM algorithm to cluster the dataset to two or three groups. Then, I use all dataset with all features to cluster the dataset into three groups: 

**Two clusters for one dimension (Petal Length):**
   Considering the scatterplots of the Iris dataset, we can classify the data into two clusters regarding the "Petal Length" dimension. In this case, Versicolor and Virginia flowers are get grouped.
We properly initialized the mean and standard deviation of each cluster. Then implemented EM algorithm. We also calculated log-likelihood. Based on the convergence of the log-likelihood we selected an appropriate number of iterations to run. 
   
**Three clusters for one dimension (Petal Width):**
  Here, three Gaussian distributions are considered to classify the dataset based on the fourth dimension which is "Petal Width". According to the dataset scatterplot, it can be suggested that Versicolor and Virginia flowers are still a bit hard to be classified. 

**Three clusters for all dimensions (the whole dataset):**
  In this case, we defined three Gaussian distributions which are initialized with proper means and standard deviations. In the implementation, the dataset is clustered based on each dimension iteratively.


--------------------------------------------

![Screen Shot 2021-06-13 at 2 55 02 PM](https://user-images.githubusercontent.com/62679750/121818041-3e0c0f00-cc5b-11eb-87a1-48ac4f661a2c.png)


