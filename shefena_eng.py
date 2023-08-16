#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow as tf
import random
import json
#from PIL import Image
import pickle
from textblob import TextBlob
#from termcolor import colored
import re
import unicodedata

# Load and preprocess data
stemmer = LancasterStemmer()

with open(r'C:\Users\WalidGebre\PycharmProjects\chapter2\MamaShefena\recipes_eng.json') as file:
    data = json.load(file)

try:
    with open(r'C:\Users\WalidGebre\PycharmProjects\chapter2\MamaShefena\dataengf.pickle', "rb") as f:
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
            doc_y.append(recette["recipe"])

        if recette["recipe"] not in labels:
                labels.append(recette["recipe"])

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

    with open(r'C:\Users\WalidGebre\PycharmProjects\chapter2\MamaShefena\dataengf.pickle', "wb") as f:
        pickle.dump((words, labels, training, output), f)

#####################################################################################

# Create and train the model
#tensorflow.reset_default_graph()
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
try:
    model.load(r'C:\Users\WalidGebre\PycharmProjects\chapter2\MamaShefena\modeleng.tflearn')
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    # Save the trained model
    model.save(r'C:\Users\WalidGebre\PycharmProjects\chapter2\MamaShefena\modelengf.tflearn')

#######################################################################################
# Define helper functions
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

def sentiment_analysis(comment):
    analysis = TextBlob(comment)
    sentiment = analysis.sentiment.polarity
    if sentiment > 0:
        return "Positive sentiment"
    elif sentiment < 0:
        return "Negative sentiment"
    else:
        return "Neutral sentiment"

#####################################################################################

def display_list(list_to_display, items_per_line=5):
    for i, item in enumerate(list_to_display, start=1):
        #formatted_item = colored(str(item), 'bold', 'green')
        print(item, end=", ")
        if i % items_per_line == 0:
            print()  # Move to the next line after printing 'items_per_line' items

#####################################################################################

def remove_special_characters(input_string):
    # Remove hyphens
    cleaned_string = input_string.replace("-", " ")

    # Remove French characters (accented characters)
    cleaned_string = ''.join(
        char for char in unicodedata.normalize('NFKD', cleaned_string) if not unicodedata.combining(char))

    return cleaned_string

#####################################################################################

# Implement the chatbot functionality

def shefena():
    print("start talking with MAMA SHEFENA to discover delightful French traditional recipes, or type 'quit' to stop!")
    allrecipes = []
    allregions = []
    allcities = []
    for recette in data["recipes"]:
        name = recette["recipe"]
        name_r = recette["region"]
        if recette["recipe"] == "kulomketematat":
            name_cities = recette["ingredients"]
            allcities.append(name_cities)
        if name_r != "mamashefena":
            allregions.append(name_r.lower())
            allrecipes.append(name.lower())
    #allrecipes = sorted(list(set(allrecipes)))
    #print(display_list(allrecipes, items_per_line=5))
    #print("===============================")
    #print(display_list(allregions, items_per_line=5))
    allcities = allcities[0]
    allcities = [c.lower() for c in allcities]
    #print(allcities)

    while True:
        inpu = input("You : ")
        inpu = remove_special_characters(inpu)

        if inpu.lower() == "":
            continue
        if inpu.lower() == "quit":
            break
        if inpu.lower() in allrecipes:
            for n in data["recipes"]:
                if n["recipe"].lower() == inpu.lower():
                    instructions = n["instructions"]
                    ingredients = n["ingredients"]
                    if inpu.lower() in ["greetings", "goodbye", "age", "name", "service", "hours", "how", "tradition"]:
                        print(random.choice(instructions))
                    else:
                        print()
                        print(f"\033[1m {inpu.upper()} : \033[0m ")
                        print()
                        print("\033[1m The following are the ingredients : \033[0m")
                        print("\n".join(["\u2022 " + item for item in ingredients]))
                        print()
                        print("\033[1m Here are the instructions : \033[0m")
                        print(("\n".join([item for item in instructions])))
                        print()
                        print("\033[3m Remember to type 'recommend' if you'd like a recommendation, and use 'comment' to share your thoughts. \033[0m")
                        break
            continue

        if inpu.lower() in allregions:
            dish_in_r = []
            for n in data["recipes"]:
                if n["region"].lower() == inpu.lower():
                    dish_n = n["recipe"]
                    dish_in_r.append(dish_n.lower())
            print("Here is a list of the dishes available in the region you provided  :")
            display_list(dish_in_r, items_per_line=5)
            print("\nChoose your preference from these available dishes.")
            user_dish = input("The dish you've selected is : ")
            #dish_in_r = [di.lower() for di in dish_in_r]
            while user_dish.lower() not in dish_in_r:
                user_dish = input("The dish you've selected is (or type 'quit') : ")
                if user_dish.lower() == "quit":
                    break
            for y in dish_in_r:
                if y.lower() == user_dish.lower():
                    for tg in data["recipes"]:
                        if tg["recipe"].lower() == y.lower():
                            instructions = tg["instructions"]
                            ingredients = tg["ingredients"]
                            print()
                            print(f"\033[1m {y.upper()} : \033[0m ")
                            print()
                            print("\033[1m The following are the ingredients : \033[0m")
                            print("\n".join(["\u2022 " + item for item in ingredients]))
                            print()
                            print("\033[1m Here are the instructions :\n \033[0m")
                            print(("\n".join([item for item in instructions])))
                            print()
                            print(
                                "\033[3m Remember to type 'recommend' if you'd like a recommendation, and use 'comment' to share your thoughts. \033[0m")
                            break
            continue

        if inpu.lower() in allcities:
            print("Enter 'recommend' and I'll provide you with a compilation of recipes from various regions.")
            continue

        if inpu.lower() == "recommend":
            docs_reg = []
            docs_rec = []
            docs_regdish = []
            for recette in data["recipes"]:
                if recette["region"] != "mamashefena":
                    docs_reg.append(recette["region"])
                docs_rec.append(recette["recipe"])
            docs_reg = sorted(list(set(docs_reg)))
            #docs_reg = i for i in docs_reg if i != "mamashefena"
            #string_to_remove = "mamashefena"
            #index_to_remove = docs_reg.index(string_to_remove)
            #docs_reg = list(docs_reg).pop(index_to_remove)
            #print("Choose a French region from the provided list, and I'll be able to suggest the available dishes for each region\n : " + ', '.join(docs_reg))
            print("Choose a French region from the provided list, and I'll be able to suggest the available dishes for each region : \n")
            display_list(docs_reg, items_per_line=5)
            print("")
            print("")
            user_region = input("The region you've selected is : ")
            user_region = remove_special_characters(user_region)
            docs_reg = [x.lower() for x in docs_reg]
            while user_region.lower() not in docs_reg:
                user_region = input("The region you've selected is (or type 'quit') : ")
                user_region = remove_special_characters(user_region)
                if user_region.lower() == "quit":
                    break
            for recette in data["recipes"]:
                if (recette["region"]).lower() == user_region.lower():
                    docs_regdish.append(recette["recipe"])
            if user_region.lower() in docs_reg:
                #print("Select one of the dishes  : " + ', '.join(docs_regdish))
                print("Select one of the dishes  :\n")
                display_list(docs_regdish, items_per_line=5)
                print("")
                print("")
                user_dishe = input("The dish you've selected is : ")
                user_dishe = remove_special_characters(user_dishe)
                docs_regdish = [di.lower() for di in docs_regdish]
                while user_dishe.lower() not in docs_regdish:
                    user_dishe = input("The dish you've selected is (or type 'quit') : ")
                    user_dishe = remove_special_characters(user_dishe)
                    if user_dishe.lower() == "quit":
                        break
                for y in docs_rec:
                    if y.lower() == user_dishe.lower():
                        for tg in data["recipes"]:
                            if tg["recipe"] == y:
                                instructions = tg["instructions"]
                                ingredients = tg["ingredients"]
                                print()
                                y_cap = y.upper()
                                print(f"\033[1m {y_cap} : \033[0m ")
                                print()
                                print("\033[1m The following are the ingredients : \033[0m")
                                print("\n".join(["\u2022 " + item for item in ingredients]))
                                print()
                                print("\033[1m Here are the instructions :\n \033[0m")
                                print(("\n".join([item for item in instructions])))
                                print()
                                print("\033[3m Remember to type 'recommend' if you'd like a recommendation, and use 'comment' to share your thoughts. \033[0m")
                                break
            continue

        if inpu.lower() == "comment":
            user_input = input("Provide your comment : ")
            sentiment_result = sentiment_analysis(user_input)
            if user_input == "":
                continue
            elif sentiment_result == "Positive sentiment":
                print("Thank you for your positive feedback!")
            elif sentiment_result == "Negative sentiment":
                print("We're sorry to hear that. Please let us know how we can improve.")
                print("Share with us your suggestions for improvement.")
                user_opinion = input("Provide your suggestions please : - ")
                if user_opinion != "":
                    print("Thank you for providing your suggestions. We will carefully consider them.")
                else:
                    continue
            else:
                print("Let's continue the conversation.")
                continue

        if inpu.lower() in ["list", "recipes", "all recipes", "recipe", "plat du jour", "dish of the day", "today's special"]:
            display_list(allrecipes, items_per_line=5)
            print("")
            continue

        else:
            results = model.predict([bag_of_words(inpu, words)])[0]
            results_index = numpy.argmax(results)
            name = labels[results_index]
            #print(results)
            #print(results[results_index])
            #print(name)

            if (results[results_index]) > 0.5:
                for tg in data["recipes"]:
                    if tg["recipe"] == name:
                        instructions = tg["instructions"]
                        ingredients = tg["ingredients"]
                # print(random.choice(responses))
                if name in ["Greetings", "Goodbye", "Age", "Name", "Service", "hours", "how", "tradition", "kulomkililat", "kulom", "kulomketematat", "kulomketema", "kulomtihztotat"]:
                    print(random.choice(instructions))
                else:
                    print()
                    name_cap = name.upper()
                    print(f"\033[1m {name_cap} : \033[0m ")
                    print()
                    print("\033[1m The following are the ingredients : \033[0m")
                    print("\n".join(["\u2022 " + item for item in ingredients]))
                    print()
                    print("Here are the instructions : \n" + ("\n".join([item for item in instructions])))
                    print()
                    print("\033[3m Remember to type 'recommend' if you'd like a recommendation, and use 'comment' to share your thoughts. \033[0m")
                    print()
            else:
                print("I'm sorry, I didn't quite catch that. Could you please try again?")

shefena()
