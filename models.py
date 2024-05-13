#  'bert-base-uncased' (english 110M)
# BASELINE MODEL  
import torch.nn as nn
from transformers import BertModel, RobertaModel

class BertBasePooledOutput(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', hidden_size=768, dropout_prob=0):
        super(BertBasePooledOutput, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1) # TODO Cambiar a 2 neuronas en vez de una

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output_d = self.dropout(pooled_output)
        logits = self.classifier(pooled_output_d) # TODO Cambiar a 2 neuronas en vez de una
        return pooled_output, logits  # TODO cambiar a logits
    
    def mixup_forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, beta):
        emb_1 = self.bert(input_ids_1, attention_mask=attention_mask_1)
        emb_2 = self.bert(input_ids_2, attention_mask=attention_mask_2)
        emb_1 = emb_1.pooler_output
        emb_2 = emb_2.pooler_output
        mixed_emb = beta * emb_1 + (1 - beta) * emb_2
        pooled_output_d = self.dropout(mixed_emb)
        logits = self.classifier(pooled_output_d)
        return mixed_emb, logits
    

# Creamos un modelo bert a partir de huggingface 'https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2'

class BertLargeCovidPooledOutput(nn.Module):
    def __init__(self, pretrained_model_name='digitalepidemiologylab/covid-twitter-bert-v2', hidden_size=1024, dropout_prob=0):
        super(BertLargeCovidPooledOutput, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output_d = self.dropout(pooled_output)
        logits = self.classifier(pooled_output_d)
        return pooled_output, logits

    def mixup_forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, beta):
        emb_1 = self.bert(input_ids_1, attention_mask=attention_mask_1)
        emb_2 = self.bert(input_ids_2, attention_mask=attention_mask_2)
        emb_1 = emb_1.pooler_output
        emb_2 = emb_2.pooler_output
        mixed_emb = beta * emb_1 + (1 - beta) * emb_2
        pooled_output_d = self.dropout(mixed_emb)
        logits = self.classifier(pooled_output_d)
        return mixed_emb, logits
    
    
class BertBaseDenseLogits(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', hidden_size=768, dense_dim=32, dropout_prob=0):
        super(BertBaseDenseLogits, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dense = nn.Linear(hidden_size, dense_dim) 
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(dense_dim, 1) 

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.bert(input_ids, attention_mask=attention_mask)
        embeddings = embeddings.pooler_output
        outputs = self.dense(embeddings)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs) 
        return embeddings, logits
    
class BertLargePooledOutput(nn.Module):
    def __init__(self, pretrained_model_name='bert-large-uncased', hidden_size=1024, dropout_prob=0):
        super(BertLargePooledOutput, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output_d = self.dropout(pooled_output)
        logits = self.classifier(pooled_output_d) 
        return pooled_output, logits 

    def mixup_forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, beta):
        emb_1 = self.bert(input_ids_1, attention_mask=attention_mask_1)
        emb_2 = self.bert(input_ids_2, attention_mask=attention_mask_2)
        emb_1 = emb_1.pooler_output
        emb_2 = emb_2.pooler_output
        mixed_emb = beta * emb_1 + (1 - beta) * emb_2
        pooled_output_d = self.dropout(mixed_emb)
        logits = self.classifier(pooled_output_d)
        return mixed_emb, logits
    
    # Creamos un modelo basado en la arquitectura Roberta de huggingface "FacebookAI/roberta-large"

class RobertaLargePooledOutput(nn.Module):
    def __init__(self, pretrained_model_name='roberta-large', hidden_size=1024, dropout_prob=0):
        super(RobertaLargePooledOutput, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output_d = self.dropout(pooled_output)
        logits = self.classifier(pooled_output_d) 
        return pooled_output, logits 

    def mixup_forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, beta):
        emb_1 = self.roberta(input_ids_1, attention_mask=attention_mask_1)
        emb_2 = self.roberta(input_ids_2, attention_mask=attention_mask_2)
        emb_1 = emb_1.pooler_output
        emb_2 = emb_2.pooler_output
        mixed_emb = beta * emb_1 + (1 - beta) * emb_2
        pooled_output_d = self.dropout(mixed_emb)
        logits = self.classifier(pooled_output_d)
        return mixed_emb, logits