import pandas as pd
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import TruncatedSVD
import numpy as np


if __name__ == "__main__":
    # Data downloading script

    ########
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('dataset.csv' not in os.listdir('../Data')):
        print('Dataset loading.')
        url = "https://www.dropbox.com/s/0sj7tz08sgcbxmh/large_movie_review_dataset.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/dataset.csv', 'wb').write(r.content)
        print('Loaded.')
    # The dataset is saved to `Data` directory
    ########

    # write your code here

    # First Stage

    df = pd.read_csv('../Data/dataset.csv')
    number_of_rows = df.shape[0]
    df.drop(index=df.query("rating >= 5 & rating <= 7").index, inplace=True)
    df["label"] = [1 if i > 7 else 0 for i in df.rating]
    df.drop(columns="rating", inplace=True)
    number_of_rows_after_filt = df.shape[0]
    prop_of_good_mov = df.query("label == 1").shape[0] / number_of_rows_after_filt
    '''
    print(number_of_rows, number_of_rows_after_filt, prop_of_good_mov, sep="\n")
    '''

    # Second Stage

    texts_train, texts_test, y_train, y_test = train_test_split(df.review.values, df.label.values, random_state=23)
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    X_train_tfidf_matrix = vectorizer.fit_transform(texts_train)
    X_test_data_collection = vectorizer.transform(texts_test)
    '''
    print(X_train_tfidf_matrix.shape[1])
    '''

    #Third Stage

    '''
    model = LogisticRegression(solver="liblinear")
    model.fit(X_train_tfidf_matrix, y_train)

    accuracy_test = accuracy_score(y_test, model.predict(X_test_data_collection))
    auc_test = roc_auc_score(y_test, model.predict_proba(X_test_data_collection)[:, 1])

    print(accuracy_test)
    print(auc_test)
    '''

    # Fourth Stage

    model = LogisticRegression(solver="liblinear", penalty="l1", C=0.15)
    model.fit(X_train_tfidf_matrix, y_train)

    '''
    accuracy_test = accuracy_score(y_test, model.predict(X_test_data_collection))
    auc_test = roc_auc_score(y_test, model.predict_proba(X_test_data_collection)[:, 1])

    print(accuracy_test, auc_test, sep="\n")
    print(len([i for i in model.coef_[0] if abs(i) > 0.0001]))
    '''
    number_of_features = len([i for i in model.coef_[0] if abs(i) > 0.0001])

    # print(model)

    # Fifth Stage

    model_fifth_stage = LogisticRegression(solver="liblinear")
    rounded_feat = np.round(number_of_features, -2)

    truncatedsvd = TruncatedSVD(n_components=rounded_feat, random_state=23)
    X_train_with_pca = truncatedsvd.fit_transform(X_train_tfidf_matrix)
    X_test_with_pca = truncatedsvd.transform(X_test_data_collection)
    model_fifth_stage.fit(X_train_with_pca, y_train)

    accuracy_test_pca = accuracy_score(y_test, model_fifth_stage.predict(X_test_with_pca))
    auc_test_pca = roc_auc_score(y_test, model_fifth_stage.predict_proba(X_test_with_pca)[:, 1])

    print(accuracy_test_pca, auc_test_pca, sep="\n")
