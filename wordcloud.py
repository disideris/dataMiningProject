import csv
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


business_list = []
film_list = []
football_list = []
politics_list = []
technology_list = []

with open('/home/dimitris/Desktop/big_data/train_set.csv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for i in reader:
        if i[4] == "Business":
            business_list.append(i)
        elif i[4] == "Film":
            film_list.append(i)
        elif i[4] == "Football":
            football_list.append(i)
        elif i[4] == "Politics":
            politics_list.append(i)
        elif i[4] == "Technology":
            technology_list.append(i)

business_list_str = ' '.join([b[3] for b in business_list])
film_list_str = ' '.join([f[3] for f in film_list])
football_list_str = ' '.join([fb[3] for fb in football_list])
politics_list_str = ' '.join([p[3] for p in politics_list])
technology_list_str = ' '.join([t[3] for t in technology_list])

my_business_stopwords = set(['said', 'will', 'say', 'including', 'one', 'one', 'still']).union(set(STOPWORDS))
my_film_stopwords = set(['said', 'will', 'say', 'see', 'one', 'much']).union(set(STOPWORDS))
my_football_stopwords = set(['said', 'will', 'say', 'see', 'one', 'us', 'time']).union(set(STOPWORDS))
my_politics_stopwords = set(['said', 'will', 'say', 'see', 'one', 'saying', 'will']).union(set(STOPWORDS))
my_tech_stopwords = set(['said', 'will', 'say', 'may', 'one', 'much', 'still']).union(set(STOPWORDS))

# Generate a word cloud image
wordcloud1 = WordCloud(max_font_size=40, stopwords=my_business_stopwords).generate(business_list_str)
plt.figure()
plt.imshow(wordcloud1, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud2 = WordCloud(max_font_size=40, stopwords=my_film_stopwords).generate(film_list_str)
plt.figure()
plt.imshow(wordcloud2, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud3 = WordCloud(max_font_size=40, stopwords=my_football_stopwords).generate(football_list_str)
plt.figure()
plt.imshow(wordcloud3, interpolation="bilinear")
plt.axis("off")
plt.show()
#
wordcloud4 = WordCloud(max_font_size=40, stopwords=my_politics_stopwords).generate(politics_list_str)
plt.figure()
plt.imshow(wordcloud4, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud5 = WordCloud(max_font_size=40, stopwords=my_tech_stopwords).generate(technology_list_str)
plt.figure()
plt.imshow(wordcloud5, interpolation="bilinear")
plt.axis("off")
plt.show()