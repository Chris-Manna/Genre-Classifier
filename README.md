# Genre Classifier
# Using Song Lyrics to Predict Genre and Hit Songs and Identify Distinctive Topics and Keywords for Each Genre
With a partner, I used Machine Learning techniques to predict the genres of 7200 songs, which songs would be hits, and keywords from genres.

In Part 1 of this blog, I walk through our EDA. In Part 2 I talk about the multiclass classification techniques used to predict song genres from lyrics. In Part 3, I walk through how we predicted hit songs from those 7200 songs. Finally, we talk about the next steps for how we might improve accuracy scores.
_____________________________________________
Part 1: Exploratory Data Analysis, (EDA)

In this process of EDA we first needed to obtain the data, then scrub and clean it, and then we can explore the data.
Obtaining Data and the initial overview:

We used two CSV’s from Kaggle. One CSV contained 300,000 songs. The other CSV contained 12,000 songs and labels for whether or not they have been on a top 100 song list.

We put these values into a pandas DataFrame where our features were the song names, band names, genre, the whole song, and whether the song was a hit or not.
Scrubbing and Cleaning Data: Grabbing the lyrics from the songs

Scrubbing the data took two days. We used Natural Language Processing tools to grab the words from the songs and clean them.

We used NLP tools like regex, we removed stopwords, we tokenized the words.

We did some Feature Engineering to find the root of each word using a process called stemmatizing and lemmatizing on the words.

After we had cleaned the dataset, we appended each word back to the Pandas DataFrame as their own column. We checked for non-values and dropped songs with non-English words or songs that didn’t contain any lyrics.

At this point, our Pandas Dataframe had 200,000 columns with 30,000 features.
Exploring Data

We chose to select eight genres from all that were listed and decided on the following genres: Rock, Pop, Hip Hop, Metal, Country, Jazz, Electronic, R&B. These genres are the target classes that we are trying to predict for each song.

Because the distribution between genres was uneven, we decided to randomly select 900 songs per genre giving us a total number of 900 songs * 8 genres = 7200 songs (rows).
_____________________________________________
Part 2: Multiclass Classification — Predicting Genre

In this section, we are going to use the features and rows we have to connect each input’s relationship to the output. We use the models.
Model Optimization: Choosing models

We wanted to see how five basic models, Multinomial Naive Bayes, Random Forest, AdaBoost, Gradient Boost, K-Nearest Neighbors, with both stemmatized and lemmatized words compare score results and pick lemmatizing over stemmatizing in our five models for future model optimization. The chart below shows our results:

![](https://github.com/Botafogo1894/Project3/blob/master/basic%205%20models.png)

We chose to go with lemmatized words over stemmatized words because every model consistently performed at least 1% better.

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

Modeling: We wanted to predict whether or not a song was on our top song hit list. We repeated the same steps as in the genre classifier model, this time creating Top 100 list of songs as our target. The target column contains a 1 for every song that was a hit and 0 to indicate songs that were not. We ran the same models and yielded the results found below and were able to predict with 96% accuracy which songs were on the top 100 song list.

![](https://github.com/Botafogo1894/Project3/blob/master/basic%205%20for%20binary%20problem.png)

Similarly to our first model, Lemmatized performed slightly better even though results were much closer this time when there were two choices.

When we used PCA on Gaussian Naive Bayes, the performance was much lower with Multinomial Naive Bayes. So, we decided to not use PCA for further optimization because it didn’t yield significant accuracy boost.

Interpreting and communicating the final results:

All of our models had around 96% score. We decided to try PCA and grid search but the results indicated there was not much room for optimization, so when it comes to binary classification it appeared that we could go with either of the Top 3 performing models and not sacrifice much accuracy.
Next Steps

In the future, to improve model accuracy scores, what I will do is introduce bigrams as new features, include more data about beats from songs, and test out neural networks.
