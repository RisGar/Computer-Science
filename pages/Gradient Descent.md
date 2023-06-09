- We can find a local minimum of a function by following its negative **gradient** or derivative
- ![_images/gradient_descent_demystified.png](https://ml-cheatsheet.readthedocs.io/en/latest/_images/gradient_descent_demystified.png)
- Each of these steps is called **learning rate**
- By finding a local minimum of a **cost function** we can find optimal values for the model
- ### Implementation
- In order to do this optimally, **Mini-batch gradient descent** is used
- We perform an update for a mini-batch with the size $n$:
- $$\theta = \theta - \eta \cdot \nabla_\theta J( \theta; x^{(i:i+n)}; y^{(i:i+n)})$$
- Common values for $n$ range between $50$ and $256$
- It is also referred to by **Stochastic gradient descent** or **SGD**, which it derives from