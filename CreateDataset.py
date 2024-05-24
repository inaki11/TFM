from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaTokenizer
import json
import nlpaug.augmenter.word as naw
import random
import nltk
from nltk.corpus import stopwords

# Ensure the stopwords are downloaded
nltk.download('stopwords')


class createAuxDataset(Dataset):
    def __init__(self, file_path, DATA_AUGMENTATION=False, MAX_LENGTH=512):
        self.data = []
        self.max_length = MAX_LENGTH
        #  'bert-base-uncased' (english 110M)
        #  'bert-large-uncased' (english 330M)
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            for item in json_data:
                text = item['text']
                category = 1 if item['category'] == 'CONSPIRACY' else 0
                self.data.append((text, category))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, category = self.data[idx]
        return text, category
"""
class createAuxDeepLDataset(Dataset):
    def __init__(self, file_path, DATA_AUGMENTATION=False, MAX_LENGTH=512, target_language='EN'):
        self.data = []
        self.max_length = MAX_LENGTH
        self.translator = deepl.Translator("aeab36e9-b204-4ec0-8984-fc5f942a9405:fx")
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            for item in json_data:
                text = item['text']
                translated = self.translator.translate_text(text, target_lang=target_language)
                category = 1 if item['category'] == 'CONSPIRACY' else 0
                self.data.append((translated, category))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, category = self.data[idx]
        return text, category

"""


class BertDataset(Dataset):
    def __init__(self, X, y, device=None, DATA_AUGMENTATION=[], WR_percentage=None, SR_percentage=None, MAX_LENGTH=512, MODEL_NAME='bert-base-uncased'):
        self.max_length = MAX_LENGTH
        #  'bert-base-uncased' (english 110M)
        #  'bert-large-uncased' (english 330M)
        self.data_augmentation = DATA_AUGMENTATION
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.X = X
        self.y = y
        
        if self.data_augmentation != []:
            if 'WR' in self.data_augmentation:
                self.WR = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute", aug_p=WR_percentage,
                                                     stopwords=stopwords.words('english') ,device=device, top_k=5)
            if 'BT' in self.data_augmentation:
                pass
            
            if 'SR' in self.data_augmentation:
                self.SR = naw.SynonymAug(aug_src='wordnet', stopwords=stopwords.words('english'), aug_p=SR_percentage, device=device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        text, label = self.X[idx], self.y[idx]
        # Aplicar aumento de datos

        if self.data_augmentation != []:
            if random.random() > 0.5:  # 50% chance to ignore data augmentation
                pass
                #print(f"original: {text}")  # Do nothing, ignore data augmentation
            else:
                if 'WR' in self.data_augmentation:
                    # apply word replacement
                    text = self.WR.augment(text)[0]
                    #print(f"WR: {text}")
                if 'BT' in self.data_augmentation:
                    #print('BT')
                    pass
                    # apply back translation
                if 'SR' in self.data_augmentation:
                    text = self.SR.augment(text)
                    # apply synonym replacement

            
        # Tokenizar el texto y obtener los input_ids
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return input_ids, attention_mask, label, text

class RobertaDataset(Dataset):
    def __init__(self, X, y, DATA_AUGMENTATION=False, MAX_LENGTH=512, MODEL_NAME='roberta-base'):
        self.max_length = MAX_LENGTH
        self.tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        text, label = self.X[idx], self.y[idx]
        # Aplicar aumento de datos
        #if DATA_AUGMENTATION != []:
            #print("aumentado de datos TO DO")
            # TO DO
        # Tokenizar el texto y obtener los input_ids
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return input_ids, attention_mask, label, text


    """
    
# Create dataset from json files and SANITIZE SPLIT


# Creamos el dataset

es_data_path = "/home/iñaki/host_data/dataset_oppositional/dataset_es_train.json"
en_data_path = "/home/iñaki/host_data/dataset_oppositional/dataset_en_train.json"

pan_es_dataset = createAuxDataset(es_data_path)
pan_en_dataset = createAuxDataset(en_data_path)


# Separate features and labels
X = [text for text, label in pan_es_dataset]
y = [label for text, label in pan_es_dataset]

# Split data using StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
    
# Save train and test data
torch.save((np.array(X_train), np.array(y_train)), 'train_es_data.pth')
torch.save((np.array(X_test), np.array(y_test)), 'test_es_data.pth')


# Separate features and labels
X = [text for text, label in pan_en_dataset]
y = [label for text, label in pan_en_dataset]

# Split data using StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
    
# Save train and test data
torch.save((np.array(X_train), np.array(y_train)), 'train_en_data.pth')
torch.save((np.array(X_test), np.array(y_test)), 'test_en_data.pth')
    
    """

    """
    
# Creamos los datasets traducidos para ampliar el training set

es_data_path = "/home/iñaki/host_data/dataset_oppositional/dataset_es_train.json"
#en_data_path = "/home/iñaki/host_data/dataset_oppositional/dataset_en_train.json"

pan_es_into_EN_dataset = createAuxDeepLDataset(es_data_path, target_language='EN-US')
#pan_en_into_ES_dataset = createAuxDataset(en_data_path)

# Separate features and labels
X = [text for text, label in pan_es_into_EN_dataset]
y = [label for text, label in pan_es_into_EN_dataset]

# Save train data
torch.save((np.array(X), np.array(y)), 'train_es_into_EN_data.pth')
    """