import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction import text
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from gensim.models import word2vec
import pandas as pd
import numpy as np
import gensim
import csv
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import time


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def featureVecMethod(words, model, num_features):
    featureVec = np.zeros(num_features, dtype="float32")
    nwords = 0

    index2word_set = set(model.wv.index2word)

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])

    featureVec = np.divide(featureVec, nwords)

    return featureVec


def getAvgFeatureVecs(contents, model, num_features):
    counter = 0
    contentFeatureVecs = np.zeros((len(contents), num_features), dtype="float32")
    for content in contents:
        contentFeatureVecs[counter] = featureVecMethod(content, model, num_features)
        counter = counter + 1

    return contentFeatureVecs


def content_wordlist(content, remove_stopwords=False):
    content_text = BeautifulSoup(content, features="html.parser").get_text()
    content_text = re.sub("[^a-zA-Z]", " ", content_text)
    words = content_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return (words)


def content_sentences(content, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(content.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(content_wordlist(raw_sentence, remove_stopwords))

    return sentences


def lemmatize(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize(token))
    return result


# English stop words
my_stop_words = text.ENGLISH_STOP_WORDS


# Classifier parameters were fine tuned with the help of sklearn's GridSearch
classifiers = [("SupportVectorMachines", SVC(C=5, kernel="linear", random_state=42)),
               ("RandomForest", RandomForestClassifier(n_estimators=100, criterion="entropy", warm_start=True, random_state=42))]

# Features list
features = ['BoW', 'SVD', 'W2V', 'BeatBench']

# Read csv files
train_data = pd.read_csv('/home/dimitris/Desktop/big_data/train_set.csv', sep='\t')
x = train_data['Content'].tolist()
x_original = x
y = train_data['Category'].tolist()

# Prepare svd reduction for later use
vectorizer = CountVectorizer(stop_words=my_stop_words)
tfidf_x = vectorizer.fit_transform(x)
# n_components was selected after testing svd.explained_variance_ratio_.sum() parameter to match 90%
svd = TruncatedSVD(n_components=3000, n_iter=5, random_state=42)
svd_x = svd.fit_transform(tfidf_x)
print(svd.explained_variance_ratio_.sum())

# Make w2v model
sentences = []
for sentence in x:
    sentences += [nltk.word_tokenize(sentence)]

sentences = []
for content in x:
    sentences += content_sentences(content, tokenizer)

model = word2vec.Word2Vec(sentences, workers=4, size=300, min_count=40, window=10, sample=1e-3)
model.init_sims(replace=True)
w2v_vectors = []

for content in x:
    w2v_vectors.append(content_wordlist(content, remove_stopwords=True))

w2v_avg_vectors = getAvgFeatureVecs(w2v_vectors, model, 300)
w2v_avg_vectors[np.isnan(w2v_avg_vectors)] = 0

# List for metric results
accuracy_results = []
precision_results = []
recall_results = []
scoring = ['accuracy', 'precision_macro', 'recall_macro']

beat_benchmark_pipe = None
flag = True

start_time = time.time()

# Perform measurements
for feature in features:
    for classifier in classifiers:

        print(classifier[0], feature)

        if feature == 'BoW':
            x = x_original
            pipe = Pipeline([('vect', CountVectorizer()), ('clf', classifier[1])])
        elif feature == 'SVD':
            x = svd_x
            pipe = Pipeline([('clf', classifier[1])])
        elif feature == 'W2V':
            pipe = Pipeline([('clf', classifier[1])])
            x = w2v_avg_vectors
        else:
            if classifier[0] == 'SupportVectorMachines':
                x2 = []
                for content in x_original:
                    x2.append(' '.join(preprocess(content)))
                x = x2
                normalizer = Normalizer()
                # Classifier parameters were fine tuned with the help of sklearn's GridSearch and default setting worked the best
                pipe = Pipeline([('vect', TfidfVectorizer()), ('scale', normalizer), ('clf', LinearSVC())])
                beat_benchmark_pipe = pipe
            else:
                break

        scores = cross_validate(pipe, x, y, scoring=scoring, cv=10, return_train_score=False, n_jobs=-1)

        accuracy_results.append(np.ndarray.mean(scores['test_accuracy']))
        precision_results.append(np.ndarray.mean(scores['test_precision_macro']))
        recall_results.append(np.ndarray.mean(scores['test_recall_macro']))

        print("Accuracy: "+str(np.ndarray.mean(scores['test_accuracy'])))
        print("Precission: "+str(np.ndarray.mean(scores['test_precision_macro'])))
        print("Recall: "+str(np.ndarray.mean(scores['test_recall_macro'])))
        print()

with open('/home/dimitris/Desktop/big_data/EvaluationMetric_10fold.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    first_row = ['Statistic Measure', 'SVM (BoW)', 'Random Forest (BoW)', 'SVM (SVD)', 'Random Forest (SVD)', 'SVM (W2V)', 'Random Forest (W2V)', 'BeatBenchmark']
    writer.writerow(first_row)
    accuracy_row = ['Accuracy', accuracy_results[0], accuracy_results[1], accuracy_results[2],
                    accuracy_results[3], accuracy_results[4], accuracy_results[5], accuracy_results[6]]
    print(accuracy_row)
    writer.writerow(accuracy_row)
    precision_row = ['Precision', precision_results[0], precision_results[1], precision_results[2],
                     precision_results[3], precision_results[4], precision_results[5], precision_results[6]]
    print(precision_row)
    writer.writerow(precision_row)
    recall_row = ['Recall', recall_results[0], recall_results[1], recall_results[2],
                  recall_results[3], recall_results[4], recall_results[5], recall_results[6]]
    print(recall_row)
    writer.writerow(recall_row)

csvFile.close()

test_data = pd.read_csv('/home/dimitris/Desktop/big_data/test_set.csv', sep='\t')
x_test = test_data['Content'].tolist()
x_test_id = test_data['Id'].tolist()

beat_benchmark_pipe.fit(x, y)
predicted = beat_benchmark_pipe.predict(x_test)

with open('/home/dimitris/Desktop/big_data/testSet_categories.csv', 'w') as csvFile:
    writer = csv.writer(csvFile, delimiter='\t')
    first_row = ['Test_Document_ID', 'Predicted_Category']
    writer.writerow(first_row)
    for id, pred in zip(x_test_id, predicted):
        new_row = [id, pred]
        print(id, pred)
        print()
        writer.writerow(new_row)

csvFile.close()

print(time.time()-start_time)

