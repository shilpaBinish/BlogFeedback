# BlogFeedback

## Description
Predict the number of comments in a blog using available features with various machine learning methods.

## Data
Dataset used here is the *BlogFeedback Data Set* by Krisztian Buza, it can be found on [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/BlogFeedback). There are 60 test files and one training file in the dataset.

In total, 281 attributes are available for each observation. Our goal is to predict the number of comments in the next 24 hours. Certain variables are removed in prepossessing due to lack of variations.

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
In the end we tried an ensemble algorithm called stacked generalization. We first picked 5 models as our "level-0 generalizers": *k*-NN, LASSO, SVM, RF, and BST. The procedure is then as follows:

1. Split the training dataset in to *k* folds.
2. For *i*-th fold of training data, train models with each of the level-0 learners on the rest of the training data.
3. Make predictions using the models we trained:
 * make predictions on *i*-th fold of data.
 * make predictions on the test set.
4. Repeat 2. and 3. until we used all *k* folds of training data.
5. Generate the level-1 data using predictions we obtained:
 * generate level-1 training data by concatenating the predictions we made on each fold of the training data.
 * generate level-1 test data by averaging the predictions we made on the test set.
6. Train a level-1 generalizer using the level-1 data and the correct responses, make predictions.

The level-1 generalizer we used in this example is elastic-net regression (&alpha;=0.2). The following table listed the performances of the level-1 test data and the stacked model:

| Model          | Error         |
| :------------- |:-------------:|
| LV-1 *k*-NN    | 0.631110      |
| LV-1 LASSO     | 0.624957      |
| LV-1 SVM       | 0.443895      |
| LV-1 RF        | 0.393018      |
| LV-1 BST       | 0.379804      |
| Stacked        | 0.382274      |

## Author
Francis Hsu, University of Illinois at Urbana–Champaign.
