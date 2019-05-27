# Predict Musical Genre
Categorize 7200 songs into 9 categories using song lyrics with 48% accuracy. 

### Obtaining Raw Data and Initial Overview:
##### Two CSV’s from Kaggle contained lyrics from songs:
- One CSV contained 300,000 songs.
- One CSV contained 12,000 songs.

##### Put the songs into a Pandas DataFrame to see features: 
- Song names
- Band names
- Genre
- Whole song

### Cleaning and Scrubbing lyrics from songs took two days:
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

In this section, we use the features and rows we have to connect each input’s relationship to the output.

# Model Optimization: Choosing models

How do the five basic models compare without money optimization?
, Multinomial Naive Bayes, Random Forest, AdaBoost, Gradient Boost, K-Nearest Neighbors, would work with both stemmatized and lemmatized words compare score results and pick lemmatizing over stemmatizing in our five models for future model optimization. 
The chart below shows our results:

![](https://github.com/Botafogo1894/Project3/blob/master/basic%205%20models.png)

We chose to go with lemmatized words over stemmatized words because, as seen in the charts, every model consistently performed at least 1% better.

We decided to go with the top three models, Multinomial Naive Bayes, Gradient Boost, and Random Forest, for further model optimization.

Next thing we did was PCA where we ran a test on our data to see how many components would preserve 80% of the variation.

Then we ran PCA with n_components = 1800 on our top three models to see if that improved performance. The graph below shows the result:

![](https://github.com/Botafogo1894/Project3/blob/master/PCA%20for%20part%201.png)

As you can see from the graph, PCA didn’t improve performance in either model, so we decided to not use PCA moving forward.

Next things we wanted to do was Grid Search on the three top-performing models and pick the model with the combination of parameters that yielded the highest accuracy score.

Grid Search on the Random Forest improved performance from 41% to 43% accuracy.

Grid Search on the Gradient Boost improved performance from 45% to 50% accuracy.

Grid Search on Naive Bayes Grid Search did not generate improved performance because the default parameters are optimal.
Interpreting and communicating the Multiclass Classification predictions results

Below you can see the graph of our top three models Final Performance after optimization.

![](https://github.com/Botafogo1894/Project3/blob/master/top%203%20models.png)

Our highest model, Gradient Boost after Grid Search yielded 50% accuracy, which is just about four times better than random guessing, 12.5%. Even though it’s not a stellar number, we are still impressed that given only 7200 lyrics we were able to predict out of eight genres with 50% accuracy.

From experimenting with Grid Search and PCA optimizations, we found that Multinomial Naive Bayes was the fastest and simplest model and it yielded only 5% less accurate than the top model.

If you have a lot of features and optimization proves to be computationally expensive, you might opt to pick Naive Bayes. If you have sufficient time and computing power and you want to optimize accuracy score, grid search with Gradient Boost is the way to go.
Part 3: Binary Classification for predicting hit songs

# Modeling: We wanted to predict whether or not a song was on our top song hit list. We repeated the same steps as in the genre classifier model, this time creating Top 100 list of songs as our target. The target column contains a 1 for every song that was a hit and 0 to indicate songs that were not. We ran the same models and yielded the results found below and were able to predict with 96% accuracy which songs were on the top 100 song list.

![](https://github.com/Botafogo1894/Project3/blob/master/basic%205%20for%20binary%20problem.png)

Similarly to our first model, Lemmatized performed slightly better even though results were much closer this time when there were two choices.

When we used PCA on Gaussian Naive Bayes, the performance was much lower with Multinomial Naive Bayes. So, we decided to not use PCA for further optimization because it didn’t yield significant accuracy boost.

Interpreting and communicating the final results:

All of our models had around 96% score. We decided to try PCA and grid search but the results indicated there was not much room for optimization, so when it comes to binary classification it appeared that we could go with either of the Top 3 performing models and not sacrifice much accuracy.

# Next Steps
In the future, to improve model accuracy scores, what I will do is introduce bigrams as new features, include more data about beats from songs, and test out neural networks.
