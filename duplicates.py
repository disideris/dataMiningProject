from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv

similarity_threshold = 0.7
similar_texts_indexes = []
article_id_list = []
article_list = []
result_set = set()

with open('/home/dimitris/Desktop/big_data/train_set.csv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for i in reader:
        article_id_list.append(i[1])
        article_list.append(i[3])

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(article_list)
cosine = cosine_similarity(tfidf_matrix)
x = np.array(cosine)

result = np.where(x >= similarity_threshold)
indexes = list(zip(result[0], result[1]))

for index in indexes:
    if index[0] != index[1]:
        if index[0] > index[1]:
            rev_index = index[::-1]
            final_tuple = rev_index, x[index[0]][index[1]]
        else:
            final_tuple = index, x[index[0]][index[1]]
        result_set.add(final_tuple)

result_list = list(result_set)
print(len(result_list))
result_list.sort(key=lambda tup: tup[1], reverse=True)

with open('/home/dimitris/Desktop/big_data/duplicatePairs.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    first_row = ['Document_ID1', 'Document_ID2', 'Similarity']
    writer.writerow(first_row)
    for row in result_list:
        new_row = [article_id_list[row[0][0]], article_id_list[row[0][1]], row[1]]
        writer.writerow(new_row)

csvFile.close()