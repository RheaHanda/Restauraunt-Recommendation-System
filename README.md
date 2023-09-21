Initially My HW3 was having a few issues, giving the RMSE real low ~0.9835. Following is the description what I did to make the Hybrid model work well.
I incorporated both item-based and model-based collaborative filtering systems in the recommendation system,
but focused more on the model-based approach as it had not been updated as frequently. 
To improve its performance, I analyzed the last iteration of the system and computed various characteristics, 
such as the minimum, maximum, and standard deviation of user ratings. I used an XGB Regressor with default hyperparameters as the model.

To enhance the system further, I looked for additional features relevant to restaurants in the "attributes" feature dictionary
of the json file, including the price range, credit card acceptance, takeout and delivery availability, reservation possibility,
and suitability for breakfast, lunch, or dinner. I believed these features could infer the overall quality of a restaurant, 
and adding them to the dataset reduced the RMSE. To find the ideal hyperparameters, 
I used the GridSearchCV package to tune the XGB Regressor by increasing the number of estimators, decreasing the learning rate, 
and lowering the max depth.

Finally, I combined the ratings from both the item-based and model-based collaborative filtering systems using static weights,
enabling us to provide more accurate and personalized recommendations to the users by leveraging the strengths of both approaches.
In summary, I enhanced the recommendation system by adding new features and tuning the hyperparameters of the XGB Regressor in the
model-based collaborative filtering system, while still incorporating the item-based approach. 
This hybrid approach enabled us to provide a better recommendation system to the users, which is crucial to the success of the project.

RMSE acheived: 0.9797884206882971

Implemented hybrid recommendation system with item-based and model-based collaborative filtering, resulting in significant system performance improvement.



Error Distribution:
>=0 and <1: 102179
>=1 and <2: 32919
>=2 and <3: 6108
>=3 and <4: 838
>=4: 0
