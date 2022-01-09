
import torch
from torch import nn 
from transformers import BertModel
from transformers.models.bert.configuration_bert import BertConfig
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np
import torch.nn.functional as F

class Encoder1(nn.Module):

    def __init__(self, pretrained, hidden_size):
        super(Encoder1, self).__init__()
    
        self.pretrained = pretrained
        self.hidden_size = hidden_size

        self.model = BertModel.from_pretrained(self.pretrained, output_hidden_states=True)
        # self.weight = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size*2, 1)

    def forward(
        self,
        input_ids=None,
        query_ids=None,
        attention_mask=None,
        que_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        document = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        query = self.model(
            query_ids,
            attention_mask=que_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        document_emb = document[1]
        query_emb = query[1]
        
        output = self.classifier(torch.cat((query_emb, document_emb), -1))
        # output = self.classifier(output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(torch.squeeze(output, 1), labels)

        return loss, output