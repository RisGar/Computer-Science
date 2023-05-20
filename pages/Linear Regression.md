- Supervised machine learning algorithm
- Predicted Output is **continuous** and has a **constant** slope ("linear")
	- For example: **Price** or **Sales**
	- **Not** Classification
- ## Simple Regression
	- Uses only one variable
	- Follows traditional "slope-intercept form":
	- $$y=mx+b$$
	- ### Predictions
		- Predictions can be made with the following function
		- $$f(w, b) = w * x + b$$
			- $w$ or $m$: Coefficient for $x$ or **Weight**
			- $x$: Independent variable or **Feature**
			- $b$: Intercept with the y-Axis or **Bias**
		- Algorithm will try to "learn" the correct weights and biases, so that we can find the **line of best fit**
		- ![_images/linear_regression_line_intro.png](https://ml-cheatsheet.readthedocs.io/en/latest/_images/linear_regression_line_intro.png)
	- ### Cost function
		- With the cost function we can optimise our weights and biases
		- For Linear Regression we can use [[Mean Squared Error]] using $mx+b$ as the prediction
		- $$MSE =  \frac{1}{N} \sum_{i=1}^{n} (y_i - (mx_i + b))^2$$
			- $N$: Total number of data points
			- $y_i$: Ground truth or **actual** label of a data point
			- $mx_i + b$: Predicted label of a data point
	- ### [[Gradient Descent]]
		- To minimise the cost we need the derivative to find which direction our parabola slope goes to
		- $$
		  f'(m,b) =
		    \begin{bmatrix}
		       \frac{1}{N} \sum -2x_i(y_i - (mx_i + b)) \\
		       \frac{1}{N} \sum -2(y_i - (mx_i + b)) \\
		    \end{bmatrix}
		  $$
	- ### Training
		- Iteratively **improving** your prediction equation
		- Looping through the dataset multiple times
			- Update weights and biases each time
		- End training when acceptable error threshold is reached
- ## Multivariable regression
	- With more features, it will become necessary to "normalise" the values to speed up calculation
	- ### Normalisation
		- Transpose the matrix to make vector math easier:
		- ```python
		  features     		 -   (200, 3)
		  features.transpose() -   (3, 200)
		  ```
		- Subtract the mean of each column (mean normalisation):
		- ```python
		  feature -= np.mean(feature)
		  ```
		- Divide each by the range of the column (feature scaling):
		- ```python
		  feature /= np.amax(feature) - np.amin(feature)
		  ```
	- ### Predictions
		- Similarly to simple regression, we use a linear function for prediction
		- $$y = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$$
	- ### Cost function
		- Replace $mx+b$ for our new function in the simple cost function
		- $$MSE =  \frac{1}{2N} \sum_{i=1}^{n} (y_i - (w_1 x_1 + w_2 x_2 + w_3 x_3 + b))^2$$
	- ### [[Gradient Descent]]
		- Similarly to simple regression we need to calculate our gradient or derivative
		- $$\begin{split}\begin{align}
		  f'(w_1) = -x_1(y - (w_1 x_1 + w_2 x_2 + w_3 x_3)) \\
		  f'(w_2) = -x_2(y - (w_1 x_1 + w_2 x_2 + w_3 x_3)) \\
		  f'(w_3) = -x_3(y - (w_1 x_1 + w_2 x_2 + w_3 x_3))
		  \end{align}\end{split}
		  $$
		-