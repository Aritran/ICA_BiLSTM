import numpy as np
import io
from sklearn.model_selection import train_test_split

filename = "word_vectors.vector"
with io.open(filename,'r',encoding='utf8') as f:
    text = f.read()
# process Unicode text
with io.open(filename,'w',encoding='utf8') as f:
    f.write(text)

list_of_words = list()


def strip_first_col(fname, delimiter=None):
    with open(fname, 'r') as fin:
        for line in fin:
            try:
                list_of_words.append(line.split(delimiter,1)[0].encode())
                yield line.split(delimiter, 1)[1]
            except IndexError:
               continue

data = np.loadtxt(strip_first_col(filename))
#data = np.loadtxt("word_vectors.vector")
print(data.shape)

for i in range(len(list_of_words)):
    list_of_words[i] = list_of_words[i].decode('ascii').lower()
Vectors = {}
for i in range(len(list_of_words)):
    Vectors[list_of_words[i]] = data[i,:]

print(Vectors['microsoft'])



def make_dataset(input_file,sentences,labels):
    with open(input_file, 'rb') as f:
        datalines = f.readlines()
        line = ""
        label = ""
        count = 0
        for data in datalines:
            try:
                data = str(data.decode('utf-8'))
                data_line = data.split('\t')
                if len(data_line) == 2:
                    if count > 30:
                        sentences.append(line)
                        labels.append(label)
                        line = ""
                        label = ""
                        count = 0
                    elif data_line[0] == 'EOS':
                        sentences.append(line)
                        labels.append(label)
                        line = ""
                        label = ""
                        count = 0
                        #corpora = corpora + '\n'
                    else:
                        if data_line[0]:
                            line = line + " " + data_line[0].lower()
                            label = label + " " + data_line[1].strip()
                        else:
                            print('{} {}'.format(data_line[0].lower(),data_line[1].strip()))
                        count = count + 1
                        #corpora = corpora + str(data_line[0])
            except Exception as e:
                print('non ascii' + str(e))
    return sentences, labels

def get_dataset():
    sentences = list()
    input_file = 'new_corpus_train.tsv'
    labels = list()

    input_file = 'new_corpus_train.tsv'
    sentences, labels = make_dataset(input_file=input_file,sentences=sentences,labels=labels)
    input_file = 'new_corpus_test.tsv'
    sentences, labels = make_dataset(input_file=input_file,sentences=sentences,labels=labels)
    num_sentences = len(sentences)
    print('labels {} {}'.format(len(labels),num_sentences))

    max_length = 0
    for sentence in sentences:
        words = sentence.split(' ')
        if len(words) > max_length:
            max_length = len(words)

    dataset = np.zeros((num_sentences,max_length,15))

    print(dataset.shape)
    i = 0
    for sentence in sentences:
        words = sentence.split(' ')
        j = 0
        for word in words:
            if word:
                #print(word)
                if word in Vectors:
                    dataset[i,j,:] = Vectors[word]
                j = j + 1
        i = i+1
    Label_Dict = {}
    Label_Dict['O'] = 1
    k = 2
    i = 0
    for label_sentence in labels:
        label_toks = label_sentence.split(' ')
        for label in label_toks:
            if label not in Label_Dict:
                Label_Dict[label] = k
                k = k+1


    label_array = np.zeros((num_sentences,max_length,k))


    label_array[:,:,0] = 1

    print('label array updated')

    i=0
    for label_sentence in labels:
        label_toks = label_sentence.split(' ')
        j = 0
        for label in label_toks:
            ind = Label_Dict[label]
            if label_array[i,j,0] ==1 :
                label_array[i,j,0] = 0
            label_array[i,j,ind] = 1
            j = j+1
        i = i+1
    data_train, data_test, labels_train, labels_test = train_test_split(dataset, label_array, test_size=0.20, random_state=42)
    return data_train, data_test, labels_train, labels_test

