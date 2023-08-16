import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
import random
import json
from PIL import Image
import pickle
from textblob import TextBlob

with open("recipes.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    doc_x = []
    doc_y = []

    for recette in data["recipes"]:
        for ingred in recette["ingredients"]:
            wrds = nltk.word_tokenize(ingred)
            words.extend(wrds)
            doc_x.append(wrds)
            doc_y.append(recette["recette"])

        if recette["recette"] not in labels:
                labels.append(recette["recette"])


    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    #####################################################################################

    training = []
    output = []
    out_empty = [0 for i in range(len(labels))]

    for x, doc in enumerate(doc_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(doc_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

#####################################################################################

#tensorflow.reset_default_graph()
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

#######################################################################################
def bag_of_words(s, words):
    bag = [0 for i in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

#####################################################################################

def shefena():
    print("start talking with the bot or type quit to stop!")
    while True:
        inpu = input("You : ")
        if inpu.lower() == "":
            continue
        if inpu.lower() == "quit":
            break
        if inpu.lower() == "recommend":
            docs_reg = []
            docs_rec = []
            docs_regdish = []
            for recette in data["recipes"]:
                if recette["region"] != "mamashefena":
                    docs_reg.append(recette["region"])
                docs_rec.append(recette["recette"])
            docs_reg = sorted(list(set(docs_reg)))
            #docs_reg = i for i in docs_reg if i != "mamashefena"
            #string_to_remove = "mamashefena"
            #index_to_remove = docs_reg.index(string_to_remove)
            #docs_reg = list(docs_reg).pop(index_to_remove)
            print("choose one of these regions : " + ', '.join(docs_reg))
            user_region = input("your choice of region : ")
            docs_reg = [x.lower() for x in docs_reg]
            while user_region.lower() not in docs_reg:
                user_region = input("your choice of region or quit : ")
                if user_region.lower() == "quit":
                    break
            for recette in data["recipes"]:
                if (recette["region"]).lower() == user_region.lower():
                    docs_regdish.append(recette["recette"])
            if user_region.lower() in docs_reg:
                print("choose one of the dishes  : " + ', '.join(docs_regdish))
                user_dishe = input("your choice of dish: ")
                docs_regdish = [di.lower() for di in docs_regdish]
                while user_dishe.lower() not in docs_regdish:
                    user_dishe = input("your choice of dish or quit: ")
                    if user_dishe.lower() == "quit":
                        break
                for y in docs_rec:
                    if y.lower() == user_dishe.lower():
                        for tg in data["recipes"]:
                            if tg["recette"] == y:
                                instructions = tg["instructions"]
                                ingredients = tg["ingredients"]
                                print()
                                print(f"\033[1m {y} : \033[0m ")
                                print()
                                print("\033[1m The ingredients are the ff : \033[0m")
                                print("\n".join(["\u2022 " + item for item in ingredients]))
                                print()
                                print("\033[1m The instruction is : \033[0m")
                                print(("\n".join([item for item in instructions])))
                                print()
                                print("\033[3m don't forget to type recommend if you want me to recommend you and write comment to say what you feel \033[0m")
                                break

        if inpu.lower() == "comment":
            def sentiment_analysis(comment):
                analysis = TextBlob(comment)
                sentiment = analysis.sentiment.polarity
                if sentiment > 0:
                    return "Positive sentiment"
                elif sentiment < 0:
                    return "Negative sentiment"
                else:
                    return "Neutral sentiment"

            user_input = input("Enter your comment: ")
            sentiment_result = sentiment_analysis(user_input)
            #print("Sentiment:", sentiment_result)

            if sentiment_result == "Positive sentiment":
                print("Thank you for your positive feedback!")
            elif sentiment_result == "Negative sentiment":
                print("We're sorry to hear that. Please let us know how we can improve.")
            else:
                print("Let's continue the conversation.")

        else:
            results = model.predict([bag_of_words(inpu, words)])[0]
            results_index = numpy.argmax(results)
            name = labels[results_index]
            #print(results)
            #print(results[results_index])
            #print(name)

            if (results[results_index]) > 0.5:
                for tg in data["recipes"]:
                    if tg["recette"] == name:
                        instructions = tg["instructions"]
                        ingredients = tg["ingredients"]
                # print(random.choice(responses))
                if name in ["salutations", "goodbye", "age", "nom", "service", "hours", "comment"]:
                    print(random.choice(instructions))
                else:
                    print()
                    print(f"\033[1m {name} : \033[0m ")
                    print()
                    print("\033[1m The ingredients are the ff : \033[0m")
                    print("\n".join(["\u2022 " + item for item in ingredients]))
                    print()
                    print("The instruction is : " + ("\n".join([item for item in instructions])))
                    print()
                    print("\033[3m don't forget to type recommend if you want me to recommend you and write comment to say what you feel \033[0m")
                    print()
            else:
                print("I didn't get that, try again.")

shefena()
