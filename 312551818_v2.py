import csv
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

train_set = "train.json"
test_set = "test.json"

with open(train_set, "r") as json_file:
    data_train = json.load(json_file)

with open(test_set, "r") as json_file:
    data_test = json.load(json_file)

def convert_json_dataframe(json):
    res = {}
    for i in range(len(json)):
        res[i] = json[i]
    return pd.DataFrame.from_dict(res, orient='index', columns=['rating', 'title', 'text', 'helpful_vote', 'verified_purchase'])

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text)

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

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
    
def lemmatize(list):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    res = []
    for item in list:
        if item[1] is None:
            res.append(item[0])
        else:
            res.append(lemmatizer.lemmatize(item[0], item[1]))
    return res
    
def nltk_pipeline(string):
    #Remove HTML tags
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

def train_model(selected_features, df_train_input, limit, tokenizer, model, model_name, from_checkpoint):
    df_train = df_train_input[selected_features]

    # CUT FOR TEST PURPOSE
    df_train = df_train.iloc[:limit]

    print("Starting NLTK pipeline on training dataset...")
    df_train['text'] = df_train['text'].apply(nltk_pipeline)    
    print("Finished NLTK pipeline on training dataset!")

    train_dataframe, validation_dataframe = train_test_split(df_train, test_size=0.33, random_state=42)
    X_train, y_train = train_dataframe['text'].to_list(), train_dataframe['rating'].astype(int).to_list()
    X_val, y_val = validation_dataframe['text'].to_list(), validation_dataframe['rating'].astype(int).to_list()

    # Reviews now range now range from 0 to 4
    y_train = [grade-1 for grade in y_train]
    y_val = [grade-1 for grade in y_val]

    # Encode features
    X_train_encoded = tokenizer(X_train, padding="max_length", truncation=True)
    X_val_encoded = tokenizer(X_val, padding="max_length", truncation=True)

    # Convert to transformers datasets
    train_revdataset = ReviewDataset(X_train_encoded, y_train)
    val_revdataset = ReviewDataset(X_val_encoded, y_val)

    training_args = TrainingArguments(output_dir=model_name, logging_dir='logs', evaluation_strategy="epoch")
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
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

def make_predictions(selected_features, df_test_input, tokenizer, model):
    df_test = df_test_input[selected_features]

    # CUT FOR TEST PURPOSE
    df_test = df_test.iloc[:]

    print("Starting NLTK pipeline on testing dataset...")
    df_test['text'] = df_test['text'].apply(nltk_pipeline)
    print("Finished NLTK pipeline on testing dataset!")

    X_test = df_test['text'].to_list()

    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, padding=True, truncation=True)

    print("Starting classification...")
    predictions = classifier(X_test)
    print("Finished classification!")

    prediction_results = extract_prediction_results(predictions)
    format_export_answers(prediction_results, selected_features)

def main():
    selected_features = ['rating'] + ['text']
    df_train = convert_json_dataframe(data_train)
    df_test = convert_json_dataframe(data_test)
    tokenizer = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
    id2label = {0: "1 star", 1: "2 stars", 2: "3 stars",3: "4 stars",4: "5 stars"}
    label2id = {"1 star": 0, "2 stars": 1,"3 stars":2,"4 stars":3,"5 stars":4}
    model = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis", num_labels=5, id2label=id2label, label2id=label2id)

    train_model(selected_features, df_train, 1250, tokenizer, model, "test_trainer", True)
    #make_predictions(selected_features, df_test, tokenizer, model)

if __name__ == '__main__':
    main()












#X_test_encoded = tokenizer(X_test, padding="max_length", truncation=True)




#test_dataset = IMDbDataset(test_encodings, test_labels)




# learning_rate=2e-05, per_gpu_train_batch_size=16, per_gpu_eval_batch_size=16, per_device_train_batch_size=1, gradient_accumulation_steps=4,gradient_checkpointing=True







"""
for result in predictions:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
"""


# Check spread of ratings

def check_ratings(dataframe):
    res = dataframe.rating.value_counts()
    print(res)
    res.plot(kind='bar')
    plt.show()