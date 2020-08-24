from xlrd import open_workbook
import numpy as np
from nltk.corpus import stopwords
import nltk

cachedStopWords = stopwords.words("english")
nltk.download('stopwords')
word_dict = {}


def read_data(xlbook= None):
    wb = open_workbook('COVID19_Dataset.xlsx')
    ind = []
    data_y = []
    data_x = []

    i = 0
    for s in wb.sheets():
        if i < 1:
        #print 'Sheet:',s.name
            for row in range(1, s.nrows):
                ind.append(row-1)
                col_names = s.row(0)
                col_value = []
                y  = (s.cell(row,0).value)
                try :
                    y = int(y)
                except :
                    pass
                data_y.append(y)
                x = (s.cell(row, 2).value)
                try:
                    x = str(x)
                except:
                    pass
                data_x.append(x)
        i = i+1
    return data_x, data_y, ind
        #col_value.append((name.value, value))
        #values.append(col_value)

#for i in range(len(data_x)):
#    print('x {} y {}'.format(data_x[i], data_y[i]))


def create_train_test(a, b, x):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    ind = int(x*len(a))
    a = a[p]
    b = b[p]
    print(len(a), len(b))
    return a[:ind], b[:ind], a[ind+1:], b[ind+1:]

def create_vec_data(train_x, train_y, test_x, test_y):
    count = 1
    training_x = []
    testing_x = []
    for i in range(train_x.shape[0]):
        txt = str(train_x[i])
        txt = txt.replace('#', ' ')
        txt = txt.replace('.', ' ')
        txt = txt.replace(',', ' ')
        txt = txt.replace(';', ' ')
        doc_vec = list()
        for word in txt.split():
            if word not in word_dict:
                word_dict[word] = count
                count = count + 1
            if word not in cachedStopWords:
                if count % 500 == 0 and count > 0:
                    doc_vec.append(0)
                else:
                    doc_vec.append(word_dict[word])
        doc_arr = np.array(doc_vec)
        training_x.append(doc_arr)
    for i in range(test_x.shape[0]):
        txt = str(train_x[i])
        txt = txt.replace('#', ' ')
        txt = txt.replace('.', ' ')
        txt = txt.replace(',', ' ')
        txt = txt.replace(';', ' ')
        doc_vec = list()
        for word in txt.split():
            if word not in word_dict and word not in cachedStopWords:
                doc_vec.append(0)
            elif word not in cachedStopWords:
                doc_vec.append(word_dict[word])
        doc_arr = np.array(doc_vec)
        testing_x.append(doc_arr)
    return training_x, train_y, testing_x, test_y, count

'''
data_x, data_y, ind = read_data()
data_x = np.array(data_x)
data_y = np.array(data_y)
print(len(data_x), len(data_y))
train_x, train_y, test_x, test_y = create_train_test(data_x, data_y, 0.7)
print(len(train_x))

training_x, training_y, testing_x, testing_y, size = create_vec_data(train_x, train_y, test_x, test_y)
training_x = np.array(training_x)
training_y = np.array(training_y)
print(size)
for i in range(len(training_x)):
    print('x {} y {}'.format(training_x[i], training_y[i]))
'''