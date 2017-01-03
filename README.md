# BlogFeedback

## Description
Predict the number of comments in a blog using available features of blog posts with different machine learning methods.

## Data
Dataset used here is the *BlogFeedback Data Set* by Krisztian Buza, it can be found on [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/BlogFeedback). There are 60 test files and one training file in the dataset.

In total, there are 281 attributes available for each observation. Our goal is to predict the number of comments in the next 24 hours. Certain variables are removed in prepossessing due to lack of variations.

## Result
The models are fitted on the log-transformed responses. Model performances are measured by mean squared error. That is, for original response *y* and prediction &fnof;&#770;(*x*), the error is calculated by (1 / *n*)&sum;[ln(1 + *y*)&minus;&fnof;&#770;(*x*)]&sup2;.

### Basic Models
We began by tuning some basic models on the dataset. Their performance are listed below:

| Model          | Error         |
| :------------- |:-------------:|
| *k*-NN         | 0.638315      |
| LASSO          | 0.635483      |
| SVM            | 0.445464      |
| Random Forest  | 0.396741      |
| Boosting       | 0.382399      |

### Weighted Average
Next we tried averaging result from different models. We trained 5 random forests and 5 boosting model. The following table listed the performances of the best of them, their averages, and the weighted average of 0.25 RF + 0.75 BST:

| Model          | Error         |
| :------------- |:-------------:|
| Best RF        | 0.396415      |
| Best BST       | 0.382642      |
| Mean RF        | 0.396803      |
| Mean BST       | 0.382934      |
| Best RF + BST  | 0.380572      |
| Mean RF + BST  | 0.381167      |

It can be seen that the weighted average model outperforms any of the individual model.

### Stacked Generalization

## Author
Francis Hsu, University of Illinois at Urbana–Champaign.
