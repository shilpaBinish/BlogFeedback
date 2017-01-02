# BlogFeedback

## Description
Predict the number of comments in a blog using available features of blog posts with different machine learning methods.

## Data
Dataset used here is the *BlogFeedback Data Set* by Krisztian Buza, it can be found on [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/BlogFeedback). There are 60 test files and one training file in the dataset.

In total, there are 281 attributes available for each observation. Our goal is to predict the number of comments in the next 24 hours. Certain variables are removed in prepossessing due to lack of variations.

## Result
The models are fitted on the log-transformed responses. Model performances are measured by mean squared error. That is, for original response *y* and prediction &fnof;&#770;(*x*), the error is calculated by (1 / *n*)&sum;[ln(1 + *y*)&minus;&fnof;&#770;(*x*)]&sup2;.

### Basic Models
We begin by tuning some basic models on the dataset. Their performance are listed below:

| Model          | Error         |
| :------------- |:-------------:|
| *k*-NN         | 0.638315      |
| LASSO          | 0.635483      |
| Random Forest  | 0.396741      |
| Boosting       | 0.382399      |

### Weighted Average

### Stacked Generalization

## Author
Francis Hsu, University of Illinois at Urbana–Champaign.
