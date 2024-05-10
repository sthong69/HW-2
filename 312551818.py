import csv
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import nltk
from sklearn.model_selection import train_test_split
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
import string as strg
import re

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from transformers import TrainingArguments, Trainer
import evaluate

# Converts JSON file to a pandas' dataframe
def convert_json_dataframe(json):
    res = {}
    for i in range(len(json)):
        res[i] = json[i]
    return pd.DataFrame.from_dict(res, orient='index', columns=['rating', 'title', 'text', 'helpful_vote', 'verified_purchase'])

# Removes HTML tags from the input string
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text)

# Replaces contracted forms to normal forms to address tokenization issues that could occur
def decontracted(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text

# Converts NLTK part-of-speech's tags to wordnet's equivalent
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
    
# Lemmatizes a sentence based on given associated part-of-speech's tags to prevent wrong lemmatization
def lemmatize(list):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    res = []
    for item in list:
        if item[1] is None:
            res.append(item[0])
        else:
            res.append(lemmatizer.lemmatize(item[0], item[1]))
    return res

# All the methods used to preprocess the text input are called in the NLTK pipeline
def nltk_pipeline(string):
    # Remove HTML tags
    res = remove_html_tags(string)
    # Remove contracted form
    res = decontracted(res)
    # Tokenize
    res = nltk.word_tokenize(res)
    # POS 
    res = nltk.pos_tag(res)
    # Convert POS to WordNet
    res = [(_[0],pos_tagger(_[1])) for _ in res]
    # Remove punctuation
    res = [_ for _ in res if _[0] not in strg.punctuation]
    # Lowercase
    res = [(_[0].lower(),_[1]) for _ in res]
    # Lemmatize
    res = lemmatize(res)
    # Remove stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    #res =  [_ for _ in res if _ not in stopwords]

    # Correct mistakes ?

    # Returns just a string
    alt_res = ""
    for word in res:
        alt_res = alt_res + word + " "

    return alt_res

# Class defined and used to convert our features' list and labels to the transformers' dataset format with appropriate methods
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Converts the string output given by our pre-trained model to a int format
def extract_prediction_results(pred):
    res = []
    for i in range(len(pred)):
        grade = 0.0
        if pred[i]['label'] == "1 star":
            grade = 1.0
        elif pred[i]['label'] == "2 stars":
            grade = 2.0
        elif pred[i]['label'] == "3 stars":
            grade = 3.0
        elif pred[i]['label'] == "4 stars":
            grade = 4.0
        elif pred[i]['label'] == "5 stars":
            grade = 5.0
        else:
            print("error")
        res.append(grade)
    return res

# Formats and exports the answers to a CSV file 
def format_export_answers(answers, list_features):
    res = input("Do you want to output the answers (Y/N)")
    if res == "N":
        return
    elif res == "Y":
        print("Formating and writing answers in csv file...")
        fields = ["index","answer"]
        rows = []
        for i in range(len(answers)):
            rows.append(["index_"+str(i), answers[i]])

        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

        with open('answer_' + str(dt_string) + "_" +str(list_features)+".csv", 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)
        print("Succesfully formated and wrote answers in a csv file !")
    else:
        format_export_answers(answers, list_features)

# All the methods used for training/fine-tuning a model are defined in the following method
def train_model(selected_features, df_train_input, limit, tokenizer, model, model_name, from_checkpoint):
    # Extract the wanted features from the dataset
    df_train = df_train_input[selected_features]
    
    # If the limit is inferior to the dataset's size, we shuffle the training dataset
    if limit != 35000:
        print("Limit inferior to 35000 : shuffling training dataset...")
        df_train = shuffle(df_train)
        print("Finished shuffling training dataset!")
    
    # Cuts the dataset to the specified size limit : used to reduce computation time when needed
    df_train = df_train.iloc[:limit]

    print("Starting NLTK pipeline on training dataset...")
    df_train['text'] = df_train['text'].apply(nltk_pipeline)
    df_train['title'] = df_train['title'].apply(nltk_pipeline)    
    print("Finished NLTK pipeline on training dataset!")

    # Splitting the dataset in order to be able to evaluate our model during training
    train_dataframe, validation_dataframe = train_test_split(df_train, test_size=0.33, random_state=42)
    X_train_title, X_train_text, y_train = train_dataframe['title'].to_list(),train_dataframe['text'].to_list(), train_dataframe['rating'].astype(int).to_list()
    X_val_title, X_val_text, y_val = validation_dataframe['title'].to_list(),validation_dataframe['text'].to_list(), validation_dataframe['rating'].astype(int).to_list()

    # Reviews now range now range from 0 to 4
    y_train = [grade-1 for grade in y_train]
    y_val = [grade-1 for grade in y_val]

    # Appends our text inputs with a "[SEP]" separator for the transformers' tokenizer later
    X_train = [X_train_title[i] + "[SEP] " + X_train_text[i] for i in range(len(X_train_title))]
    X_val = [X_val_title[i] + "[SEP] " + X_val_text[i] for i in range(len(X_val_title))]

    # Encodes features with the given tokenizer, pads and truncates the content to maximum length for the model's input
    X_train_encoded = tokenizer(X_train, padding="max_length", truncation=True)
    X_val_encoded = tokenizer(X_val, padding="max_length", truncation=True)

    # Convert to transformers' compatible datasets
    train_revdataset = ReviewDataset(X_train_encoded, y_train)
    val_revdataset = ReviewDataset(X_val_encoded, y_val)

    # Defines the training's arguments
    training_args = TrainingArguments(output_dir=model_name, logging_dir='logs', evaluation_strategy="epoch", num_train_epochs=11)
    
    # Defines our way to compute our model's accuracy during training 
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    # Defines our tranformers' trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_revdataset,
        eval_dataset=val_revdataset,
        compute_metrics=compute_metrics,
    )

    print("Starting model training...")
    trainer.train(resume_from_checkpoint=from_checkpoint)
    print("Finished model training! Saving...")
    trainer.save_model(output_dir="trained-models/"+model_name)
    print("Model saved!")

# All the methods used for making predictions on the test dataset with a given model are defined in the following method
def make_predictions(selected_features, df_test_input, tokenizer, model):
    # Extract the wanted features from the dataset
    df_test = df_test_input[selected_features]

    # CUT FOR TEST PURPOSE
    df_test = df_test.iloc[:]

    print("Starting NLTK pipeline on testing dataset...")
    df_test['text'] = df_test['text'].apply(nltk_pipeline)
    df_test['title'] = df_test['title'].apply(nltk_pipeline)  
    print("Finished NLTK pipeline on testing dataset!")

    X_test_title, X_test_text = df_test['title'].to_list(),df_test['text'].to_list()
    X_test = [X_test_title[i] + "[SEP] " + X_test_text[i] for i in range(len(X_test_title))]

    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, padding=True, truncation=True)
 
    print("Starting classification...")
    predictions = classifier(X_test)
    print("Finished classification!")

    prediction_results = extract_prediction_results(predictions)
    format_export_answers(prediction_results, selected_features)

def main():
    train_set = "train.json"
    test_set = "test.json"

    with open(train_set, "r") as json_file:
        data_train = json.load(json_file)

    with open(test_set, "r") as json_file:
        data_test = json.load(json_file)

    selected_features = ['rating'] + ['title'] + ['text']
    df_train = convert_json_dataframe(data_train)
    df_test = convert_json_dataframe(data_test)
    tokenizer = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
    id2label = {0: "1 star", 1: "2 stars", 2: "3 stars",3: "4 stars",4: "5 stars"}
    label2id = {"1 star": 0, "2 stars": 1,"3 stars":2,"4 stars":3,"5 stars":4}
    
    model = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis", num_labels=5, id2label=id2label, label2id=label2id)
    trained_model = AutoModelForSequenceClassification.from_pretrained("trained-models/35000_3_alltext", num_labels=5, id2label=id2label, label2id=label2id)

    #train_model(selected_features, df_train, 100, tokenizer, model, "test_trainer2", False)
    make_predictions(selected_features, df_test, tokenizer, trained_model)

if __name__ == '__main__':
    main()

# Checks spread of ratings
def check_ratings(dataframe):
    res = dataframe.rating.value_counts()
    print(res)
    res.plot(kind='bar')
    plt.show()