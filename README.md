# ML Matrix Factorization recommender
Implementation of the [winning recommender system from the Netflix competition](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf).
Uses matrix decomposition to derive a P and Q matrix which can be used to make predictions.  
Uses gradient descent to arrive at the solution.  

## How to Use.

Predict using the ProductRecommender Class.   
Can be used in two ways:   
   
1. First: To recommend using only a users x product matrix along with a k value for # of features to discover.   
```python
    #import predictor
    from recommender.matrix_factor_model import ProductRecommender
    
    # fake data. (2 users, three products)
    user_1 = [1, 2, 3]
    user_2 = [0, 2, 3]
    data = [user_1, user_2]
    
    # train model
    modelA = ProductRecommender()
    modelA.fit(data)
    
    # predict for user 2 
    modelA.predict_instance(1)
    # prints array([ 0.9053102 ,  2.02257811,  2.97001565])
```   
   
2. Second: To recommend when you want to provide a list of features for movies and only learn P (user -> feature strength).
```python
    #import predictor
    from recommender.matrix_factor_model import ProductRecommender
    
    # fake data. (2 users, three products)
    user_1 = [1, 2, 3]
    user_2 = [0, 2, 3]
    data = [user_1, user_2]
    
    # product features (year made, height)
    product_1 = [2014, 74]
    product_2 = [2016, 89]
    Q = [product_1, product_2]
    
    # train model passing in Q
    modelB = ProductRecommender()
    modelB.fit(data, Q)
    
    # predict for user 2 
    modelB.predict_instance(1)
    # prints array([ 0.9053102 ,  2.02257811,  2.97001565])
```   
   
## Defaults   
|parameter   |default value   |description   |
|---|---|---|
|user_x_product   |Must provide   |User x product Matrix   |
|product_x_features   |Optional (null)   |Products_x_features matrix. Sets latent_features_guess to number of features here. |
|latent_features_guess   |2   |Features we want to learn   |
|learning_rate   |0.0002   |Size of learning steps   |
|steps   |5000   |Max number of steps until convergence   |
|regularization_penalty   |0.02   |Penalty for over/under fitting   |
|convergeance_threshold   |0.001   |Error amount to terminate (we solved the problem). Otherwise uses steps.   |  
    
## License  
MIT  

## Author  
William Falcon  
