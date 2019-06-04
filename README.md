# Predict Musical Genre
Categorize 7200 songs into eight distinct genres using song lyrics with 48% accuracy. 

### Obtaining Raw Data and Initial Overview:
##### Two CSV’s from Kaggle contained lyrics from songs:
- One CSV contained 300,000 songs.
- One CSV contained 12,000 songs.
Only used songs that contained lyrics and English characters

##### Put the songs into a Pandas DataFrame to see features: 
- Song names, Band names, Genre, Whole song

### Scrubbing lyrics from songs:
Tools used: NLTK, remove stopwords, Stemmatizing and Lemmatizing
(Look up which library used to remove stopwords)

- Append words to the Pandas DataFrame as their own feature. 
- Drop non-values, songs with non-English characters, and songs not containing lyrics.
- Pandas Dataframe has 200,000 rows with 30,000 features.
[]()

Selected eight distinct genres to categorize songs into
- Rock, Pop, Hip Hop, Metal, Country, Jazz, Electronic, R&B. 
Target classes that we are predicting for each song.

Because the distribution between genres was uneven, we decided to randomly select 900 songs per genre 
Total number of 900 songs * 8 genres = 7200 songs (rows).

![]()

# Multiclass Classification  —  Predicting Genre

### Choosing Models

How do the five basic models compare without money optimization?
Multinomial Naive Bayes, Random Forest, AdaBoost, Gradient Boost, K-Nearest Neighbors, would work with both stemmatized and lemmatized words compare score results and pick lemmatizing over stemmatizing in our five models for future model optimization. 
The chart below shows our results:

![](https://github.com/Botafogo1894/Project3/blob/master/basic%205%20models.png)

- Each model consistently performed at least 1% better when we use the lemmatize function

We decided to go with the top three models for further model optimization:
- Multinomial Naive Bayes
- Gradient Boost
- Random Forest

##### Performed PCA 
How many components would preserve 80% of the variation.

Then we ran PCA with n_components = 1800 on our top three models to see if that improved performance. 
The graph below shows the result:

![](https://github.com/Botafogo1894/Project3/blob/master/PCA%20for%20part%201.png)

- PCA didn’t improve performance in models. 

Performed Grid Search on the three top-performing models and picked the model with the combination of parameters that yielded the highest accuracy score.

Below you can see the graph of our top three models Final Performance after optimization.

![](https://github.com/Botafogo1894/Project3/blob/master/top%203%20models.png)
- Grid Search on the Random Forest improved performance from 41% to 43% accuracy.

- Grid Search on the Gradient Boost improved performance from 45% to 50% accuracy.

- Grid Search on Naive Bayes Grid Search did not generate improved performance because the default parameters are optimal

Our highest model, Gradient Boost after Grid Search yielded 50% accuracy, which is just about four times better than random guessing, 12.5%. Even though it’s not a stellar number, we are still impressed that given only 7200 lyrics we were able to predict out of eight genres with 50% accuracy.

From experimenting with Grid Search and PCA optimizations, we found that Multinomial Naive Bayes was the fastest and simplest model and it yielded only 5% less accurate than the top model.

### Interpreting
If you have a lot of features and optimization proves to be computationally expensive, you might opt to pick Naive Bayes. If you have sufficient time and computing power and you want to optimize accuracy score, grid search with Gradient Boost is the way to go.

# Next Steps
In the future, to improve model accuracy scores, what I will do is introduce bigrams as new features, include more data about beats from songs, and test out neural networks.
