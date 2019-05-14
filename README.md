# Genre Classifier
# Using Song Lyrics to Predict Genre and Hit Songs and Identify Distinctive Topics and Keywords for Each Genre
Categorize 7200 songs into 9 categories based only on lyrics.

Exploratory Data Analysis, (EDA)
Obtaining Data and Initial Overview:

We used two CSV’s that contained songs with lyrics from Kaggle. 
- One CSV contained 300,000 songs.
- One CSV contained 12,000 songs.

We put the songs into a Pandas DataFrame. 
- Our features were the song names, band names, genre, the whole song, and whether the song was a hit or not.

Gathering the lyrics from the songs
- Scrubbing the data took two days.
Tools used: NLTK, REGEX, various tokenization techniques, imported libraries to remove stopwords.

#Feature Engineering
To find the root of each word use the process called stemmatizing and lemmatizing on the words.
https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
Stemming usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes. 
Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma.

After we cleaned the dataset, we appended each word back to the Pandas DataFrame as their own feature. We checked for non-values and dropped songs with non-English words or songs that didn’t contain any lyrics.

At this point, our Pandas Dataframe had 200,000 rows with 30,000 features.
[]()

We chose to select eight distinct genres: Rock, Pop, Hip Hop, Metal, Country, Jazz, Electronic, R&B. 
These genres are the target classes that we are trying to predict for each song.

Because the distribution between genres was uneven, we decided to randomly select 900 songs per genre giving us a total number of 900 songs * 8 genres = 7200 songs (rows).
_____________________________________________
Part 2: Multiclass Classification  —  Predicting Genre

In this section, we use the features and rows we have to connect each input’s relationship to the output.
Model Optimization: Choosing models

We wanted to see how five basic models, Multinomial Naive Bayes, Random Forest, AdaBoost, Gradient Boost, K-Nearest Neighbors, would work with both stemmatized and lemmatized words compare score results and pick lemmatizing over stemmatizing in our five models for future model optimization. 
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

Modeling: We wanted to predict whether or not a song was on our top song hit list. We repeated the same steps as in the genre classifier model, this time creating Top 100 list of songs as our target. The target column contains a 1 for every song that was a hit and 0 to indicate songs that were not. We ran the same models and yielded the results found below and were able to predict with 96% accuracy which songs were on the top 100 song list.

![](https://github.com/Botafogo1894/Project3/blob/master/basic%205%20for%20binary%20problem.png)

Similarly to our first model, Lemmatized performed slightly better even though results were much closer this time when there were two choices.

When we used PCA on Gaussian Naive Bayes, the performance was much lower with Multinomial Naive Bayes. So, we decided to not use PCA for further optimization because it didn’t yield significant accuracy boost.

Interpreting and communicating the final results:

All of our models had around 96% score. We decided to try PCA and grid search but the results indicated there was not much room for optimization, so when it comes to binary classification it appeared that we could go with either of the Top 3 performing models and not sacrifice much accuracy.
Next Steps

In the future, to improve model accuracy scores, what I will do is introduce bigrams as new features, include more data about beats from songs, and test out neural networks.
