import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
from PIL import Image
import pickle

with open(r'C:\Users\WalidGebre\PycharmProjects\chapter2\CHATBOT\recipes.json') as file:
    data1 = json.load(file)
try:
    with open("data1.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data1["recipes"]:
        for pattern in intent["ingredients"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["name"])

        if intent["name"] not in labels:
            labels.append(intent["name"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data1.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)

#tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)



try:
    model.load("model.tflearn1")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn1")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    print("You can also type recommend if you want me to recommend you")
    while True:
        inp = input("You: ")
        if inp.lower() == "recommend":
            docs_r = []
            docs_n = []
            docs_regdish = []
            for intent in data1["recipes"]:
                docs_r.append(intent["region"])
                docs_n.append(intent["name"])
            print("choose one of these regions : " + ', '.join(docs_r))
            user_region = input()
            for intent in data1["recipes"]:
                if (intent["region"]).lower() == user_region.lower():
                    docs_regdish.append(intent["name"])
            print("choose one of the dishes : " + ', '.join(docs_regdish))
            user_dishe = input()
            for y in docs_n:
                if y.lower() == user_dishe.lower():
                    for tg in data1["recipes"]:
                        if tg["name"] == y:
                            instructions = tg["instructions"]
                            ingredients = tg["ingredients"]
                            cooking_time = tg["cooking_time"]
                            difficulty_level = tg["difficulty_level"]
                    # print(random.choice(responses))
                    print(f"\033[1m {y} : \033[0m ")
                    print()
                    print("\033[1m The ingredients are the ff : \033[0m")
                    print("\n".join(["\u2022 " + item for item in ingredients]))
                    print()
                    print("\033[1m The instruction is : \033[0m" + ("\n".join([item for item in instructions])))
                    print()
                    print("\033[1m The Cooking time is : \0330m" + cooking_time)
                    print()
                    print("\033[1m Difficulty level : \033[0m" + difficulty_level)
                    print()
                    print("\033[3m don't forget to type recommend if you want me to recommend you \033[0m")
                    break

        if inp.lower() == "quit":
            break
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        name = labels[results_index]
        #print(results)
        #print(results[0][results_index])

        if (results[0][results_index]) > 0.5:
            for tg in data1["recipes"]:
                if tg["name"] == name:
                    instructions = tg["instructions"]
                    ingredients = tg["ingredients"]
                    cooking_time = tg["cooking_time"]
                    difficulty_level = tg["difficulty_level"]

            #print(random.choice(responses))
            if name in ["salutations", "goodbye", "age", "nom", "service", "comment"]:
                print(random.choice(instructions))
            else:
                print(f"\033[1m {name} : \033[0m ")
                print()
                print("\033[1m The ingredients are the ff : \033[0m")
                print("\n".join(["\u2022 " + item for item in ingredients]))
                print()
                print("The instruction is : " + ("\n".join([item for item in instructions])))
                print()
                print("The Cooking time is : " +  cooking_time)
                print()
                print("\033[1m Difficulty level : \033[0m" + difficulty_level)
                print()
                print("\033[3m don't forget to type recommend if you want me to recommend you \033[0m")
        else:
            print("I didn't get that, try again.")
chat()

