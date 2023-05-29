import cv2
import nltk.corpus
import random
import numpy as np


sent = nltk.corpus.brown.sents()
n_grams = {}
for sen in sent:
    words = [word for word in sen if word[0].isalpha()]
    for ix in range(len(words)-1):
        try:
            n_grams[words[ix]].append(words[ix+1])
        except KeyError as _:
            n_grams[words[ix]] = []
            n_grams[words[ix]].append(words[ix+1])


def objects(img):
    config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = 'frozen_inference_graph.pb'
    model = cv2.dnn_DetectionModel(frozen_model, config_file)
    class_labels = []
    file_name = 'labels.txt'
    with open(file_name, 'rt') as fpt:
        class_labels = fpt.read().rstrip('\n').split('\n')
    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)
    img = cv2.imread(img)
    ClassIndex, confidece, bbox = model.detect(img, confThreshold=0.5)
    list_all = []
    for i in range(len(ClassIndex)):
        for x in range(len(class_labels)):
            if x == int(ClassIndex[i]) - 1:
                list_all.append(class_labels[x])
    return list_all


def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def gen_sent( word, nb = 7):
    global n_grams
    next_word = random.choice(list(n_grams.keys()))
    word.append(next_word)
    for t in range(nb+len(word)):
        next_word = random.choice(n_grams[next_word])
        word.append(next_word)
        return " ".join(word)


def pri(x):
    img = cv2.imread("grey.png")
    org = np.shape(img)
    org1 = (0, np.shape(img)[0] + 100)
    img_n = []
    for i in range(org[0] + 150):
        c = []
        for j in range(org[1]):
            if i < org[0]:
                c.append(img[i][j])
            else:
                c.append([232, 232, 232])
        img_n.append(c)
    img_n = np.array(img_n)
    img_n = img_n.astype(np.uint8)
    img_n = cv2.putText(img=img_n, text=x, org=org1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0), thickness=2)
    cv2.imshow("New Image", img_n)
    cv2.waitKey(0)


pri(gen_sent( unique(objects("grey.png")), nb =8))

