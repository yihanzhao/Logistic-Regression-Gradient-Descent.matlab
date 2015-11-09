# logistic_regression_gradient_descent_matlab
First, I randomly give two vectors of logistic model parameters, called "theta old" and "theta new". Then, I get the first gradient of the empirical loss function for the "theta old",
gradient= 1( 1−yi − yi )f′(xi;θ) N 1−f(xi;θ f(xi;θ
where f(x; θ) = (1 + exp(−θT X))−1
Then, I updated "theta new" to be "theta new=theta old-alpha*gradient".
I continue iterating the above steps untill I get a new gradient that minus the older one less
than a specific number, which is called tolerance.
To be specific, tolerance and alpha can be set arbitrarily. The following figure is
drawn when alpha=0.05, and tolerance=0.01. From the decision boundary, we can see there are 2 points misclassified when X1 and X2 are both around 0.25.Thus, the misclassification rate is 2 out of 200 in this case.
