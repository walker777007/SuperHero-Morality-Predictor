<p align="center">
<img src="images/Sentinel.jpg" width="700" height="475">
</p>

# The Sentinel (Super Hero Morality Predictor)
**Predicting whether comic book characters are good or evil**
<br>Walker Stevens
\
[Linkedin](https://www.linkedin.com/in/walker-stevens-31783087/) | [Github](https://github.com/walker777007)
\
[Slides](https://docs.google.com/presentation/d/1P7B_7BOsmnOh5gjEXYo_EOSriYO3FBwkCnnmfi7P5f4/edit?usp=sharing)

## Table of Contents

* [Motivation](#motivation)
* [Data Exploration](#data-exploration)
  * [Pipeline](#pipeline)
  * [Moral Inequality](#moral-inequality)
* [Machine Learning Modeling](#machine-learning-models)
  * [Scoring Comparisons](#scoring-comparisons)
  * [Best Model](#best-model)
* [Conclusion](#conclusion)

## Motivation

In the Marvel comics, there are robots called [Sentinels](https://marvel.fandom.com/wiki/Sentinels), who hunt down [Mutants](https://marvel.fandom.com/wiki/Mutants) (super-humans).  My idea is "What if we could create a model that could predict whether a comic book character was good or evil based on all the known information about them?"  Essentially, can we create a new type of Sentinel that has a moral compass when it hunts down superheros/villains?  Giving robots the ability to decide who is good or evil can't go wrong, right?

## Data exploration

### Pipeline

Where I got the data:
* DC Characters Info: [DC Wikia](https://dc.fandom.com/wiki/DC_Comics_Database)
* Marvel Characters Info : [Marvel Wikia](https://marvel.fandom.com/wiki/Marvel_Database/)

In an [article Fivethirtyeight wrote in 2014](https://fivethirtyeight.com/features/women-in-comic-books/), they scraped both these wikis and compiled them into two separate csv files which they now host on their [Github](https://github.com/fivethirtyeight/data/tree/master/comic-characters).

Features that they scraped about each character:
* **Alignment**: Whether the character is Good, Bad or Neutral
* **ID**: Whether they have a public, secret identity, etc.
* **Eye Color**
* **Hair Color**
* **Sex**
* **GSM**: Gender or Sexual Minority
* **Alive**: Whether the character is currently alive or dead
* **Appearances**: The number of appearances the character has had in comics
* **Year**: The year the character was introduced

Once all the data CSV files were collected, I used pandas in order to group them into dataframes, and proceeded to do all my calculations and tests after.  I categorized bad characters as 1 and good characters as 0, so predicting a bad character correctly would be considered a true positive.

### Moral inequality

Before going on to modeling, I did some exploratory data analysis to see which features were most correlated with a character being good or bad.  What I found is that there is quite the "moral inequality" when it comes to certain characteristics.

As we can see below, the most popular (most appearances) characters tend to be good.  The top 50 good characters all have more appearances than the most popular bad character.
<p align="center">
<img src="plots/Appearances_by_Morality.png" width="800" height="550">
</p>

Another discrepancy in good vs evil characters is their sex.  Bad characters tend to be overwhelmingly male and female characters tend to be good.
<p align="center">
<img src="plots/Morality_by_Sex.png" width="800" height="550">
</p>

In terms of physical characteristics, there is also a morality imbalance.  Characters who have red eyes are disproportionately evil whereas blue eyed characters are mostly good.  As well, there seems to be quite the bias against bald characters, as they tend to be more evil as well.
<p align="center">
<img src="plots/Eye_Color_Morality.png" width="800" height="381">
<img src="plots/Hair_Color_Morality.png" width="800" height="381">
</p>

## Machine Learning Modeling

To start I separated the data into a training set and a test (holdout) set.  As well, I chose a host of supervised learning models to classify the data:
* [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree)
* [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
* [K Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
* [Support Vector Machine](https://en.wikipedia.org/wiki/Support-vector_machine)
* [Random Forest](https://en.wikipedia.org/wiki/Random_forest)
* [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)
* [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
* [XGBoost](https://en.wikipedia.org/wiki/XGBoost)

### Scoring Comparisons

For my scores, I decided to compare Accuracy, Precision and Recall.  Using the means of a 5 Fold Cross Validation, I got the scores below.
<p align="center">
<img src="plots/Model_Scores.png" width="800" height="381">
</p>
As well, in the ROC Curve, we can see which models have the best diagnostic abilities.
<p align="center">
<img src="plots/ROC.png" width="800" height="550">
</p>
XGBoost and Gradient Boost are the 2 best performing models, as they have the best accuracy and precison and only a marginally smaller recall than AdaBoost and KNN.  Choosing between the two was a tossup considering how similar their results were, but I went with XGBoost since it was ever so higher on the ROC Curve.

### Best Model

After tuning my final model, the hyperparameters were:
n_estimaors (Number of gradient boosted trees) = 500
learning_rate (Rate to shrink the contribution of each tree) = 0.07
max_depth (Maximum tree depth for base learners) = 4
Using the training data, the feature importances (calculated by using the gain) of the XGBoost model were:
<p align="center">
<img src="plots/XGBoost_Feature_Importances.png" width="800" height="381">
</p>
As expected, certain features like whether the character is female or has no hair, or red eyes were important in the classification.  On the test data, our model had an accuracy of <strong>71.8%</strong>, which was actually a bit better than our training accuracy of <strong>70.0%</strong>. The confusion matrix on the test data shows that it has a higher recall than negative predicted value, which means it is predicting evil characters better than good characters.  Unfortunately this means a lot of innocent good characters will be killed, I guess our sentinel can't entirely shrug off its intrinsic programming.
<p align="center">
<img src="plots/XGBoost_Confusion_Matrix.png" width="800" height="550">
</p>
When we consider neutral characters as well as good and bad characters, our model performs much worse.  I guess our sentinel has a very black and white sense of morality.
<p align="center">
<img src="plots/XGBoost_Confusion_Matrix_Neutral.png" width="800" height="550">
</p>
