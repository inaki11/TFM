#  'bert-base-uncased' (english 110M)
# BASELINE MODEL  
import torch.nn as nn
from transformers import BertModel

class BertBasePooledOutput(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', hidden_size=768, dropout_prob=0):
        super(BertBasePooledOutput, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probabilities = self.sigmoid(logits)
        return probabilities