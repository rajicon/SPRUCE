#this makes classes that inherit from bertram, but try my techniques
#since we inherit, hopefully most of the other bertram stuff is usable out of the box

import bertram
from bertram import Bertram, BertramConfig, requires_context, requires_form, requires_sep, requires_shallow, MODE_FORM, MODE_CONTEXT, MODE_SHALLOW, MODE_REPLACE, MODE_ADD, MODE_ADD_QUOTES, MODES, ReliabilityMeasure

import os

from typing import Callable, List, Dict
import jsonpickle
import torch
import torch.nn as nn
from torch.nn import MSELoss, Module, Embedding
from transformers import BertModel, BertConfig, RobertaModel, BertTokenizer, RobertaTokenizer, BertPreTrainedModel, \
    RobertaConfig, PreTrainedTokenizer, PreTrainedModel

import log
from input_processor import InputProcessor
from ngram_models import BagOfNgrams
from utils import length_to_mask
import torch.nn.functional as F

from HiCE_Transformer_Methods import EncoderLayer, LayerNorm, MultiHeadedAttention, MultiHeadedAttention, SublayerConnection, PositionalEncoding, PositionalAttention, CharCNN, CrossEncoderLayer


class Spruce(Bertram):
    """for testing inheritance."""
    
    base_model_prefix = "spruce"


    def __init__(self, transformer_config: BertConfig, bertram_config: BertramConfig, do_setup=False):

        super(Spruce, self).__init__(transformer_config, bertram_config, do_setup)

        num_layers = bertram_config.variant_num_layers
        transformer_dropout = bertram_config.transformer_dropout

        #set at 12 layers to match bert, probably not necessary
        self.sub_self_att = nn.ModuleList([CrossEncoderLayer(12, 768, att_dropout = transformer_dropout, ffn_dropout = transformer_dropout, res_dropout = transformer_dropout) for _ in range(12)])


        self.cc_self = nn.ModuleList([CrossEncoderLayer(12, 768, att_dropout = transformer_dropout, ffn_dropout = transformer_dropout, res_dropout = transformer_dropout) for _ in range(num_layers)])
        self.ss_self = nn.ModuleList([CrossEncoderLayer(12, 768, att_dropout = transformer_dropout, ffn_dropout = transformer_dropout, res_dropout = transformer_dropout) for _ in range(num_layers)])
        
        
        self.final_combiner = nn.Linear(4*bertram_config.output_size, bertram_config.output_size)

        self.gate_combiner = bertram_config.gate_combiner
       
        self.two_combiner1 = nn.Linear(2*bertram_config.output_size, 1)
        self.two_combiner2 = nn.Linear(2*bertram_config.output_size, 1)
        self.two_combiner3 = nn.Linear(2*bertram_config.output_size, 1)
        
        if self.gate_combiner == 'hierarchy_gate_with_freqs':
            self.two_combiner4 = nn.Linear(2*bertram_config.output_size+4, 1) #num sent, max subcount, avg sub count, min sub count

        self.four_combiner = nn.Linear(4*bertram_config.output_size, 4)



    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                nr_of_contexts: torch.Tensor,
                mask_positions: torch.Tensor,
                attention_mask: torch.Tensor,
                ngram_ids: torch.Tensor,
                ngram_lengths: torch.Tensor,
                target_vectors: torch.Tensor = None):

        """
        Process a batch of words and contexts and generate embeddings. If `target_vectors` is not `None`,
        the loss is returned. Otherwise, the BERTRAM embeddings for all given words are returned.
        :param input_ids:           tensor of input token ids
        :param token_type_ids:      tensor of token type ids
        :param nr_of_contexts:      list of context lengths per word
        :param mask_positions:      tensor of shape sum(nr_of_contexts), containing the positions of the [MASK]
                                    tokens in the given contexts (assuming one per line)
        :param attention_mask:      attention mask tensor for the underlying transformer language model
        :param ngram_ids:           tensor of ngram ids for each word
        :param ngram_lengths:       list of ngram lengths (i.e., number of ngrams per word)
        :param target_vectors:      tensor containing the target vectors for each word (optional)
        """

        if not self.is_setup:
            raise ValueError("setup() must be called before using the model.")

        # if input has an additional 0th dimension with only one entry, it means we are in data parallel mode
        # and must first remove this additional dimension
        data_parallel_mode = input_ids is not None and len(input_ids.shape) == 3

        if data_parallel_mode:
            input_ids = torch.squeeze(input_ids, 0)
            token_type_ids = torch.squeeze(token_type_ids, 0)
            nr_of_contexts = torch.squeeze(nr_of_contexts, 0)
            mask_positions = torch.squeeze(mask_positions, 0)
            attention_mask = torch.squeeze(attention_mask, 0)
            ngram_ids = torch.squeeze(ngram_ids, 0)
            ngram_lengths = torch.squeeze(ngram_lengths, 0)
            target_vectors = torch.squeeze(target_vectors, 0)

        output_vectors = None
        ngram_vectors = None

        if requires_form(self.bertram_config.mode):
            ngram_vectors, ngram_embs, ngram_mask = self.ngram_processor.forward_ngram_all_embs(ngram_ids, ngram_lengths)
            if input_ids is None:
                #return ngram_vectors
                ngram_mask = ngram_mask.unsqueeze(-2).unsqueeze(-2)

                for layer in self.sub_self_att:
                    ngram_embs = layer(ngram_embs, ngram_embs, mask=ngram_mask)

                ss_self = ngram_embs
                for layer in self.ss_self:
                    ss_self = layer(ss_self, ngram_embs, mask=ngram_mask)
                            
                ngram_mask = torch.reshape(ngram_mask, (ngram_mask.size()[0], ngram_mask.size()[3]))

                ngram_mask = ngram_mask.unsqueeze(-1)
            
                ss_mean = torch.sum(ss_self * ngram_mask, dim=1) / torch.sum(ngram_mask, dim=1)
                #sub_self = ngram_vectors
                
                sub_self = torch.sum(ngram_embs * ngram_mask, dim=1) / torch.sum(ngram_mask, dim=1)

                
                a = torch.cat([sub_self, ss_mean], dim=1)
                alpha_sub = self.two_combiner1(a)
                alpha_sub = torch.sigmoid(alpha_sub)
                final_sub_out = (alpha_sub * sub_self) + ((1-alpha_sub) * ss_mean)
                
                return final_sub_out
        
                
                

        if self.bertram_config.mode == MODE_FORM:
            output_vectors = ngram_vectors
            final_out = output_vectors
            

        if requires_context(self.bertram_config.mode):
            overwrite_fct = None
            if self.bertram_config.mode == MODE_REPLACE:
                overwrite_fct = self.get_mask_oef(ngram_vectors, nr_of_contexts, mask_positions)
            elif requires_sep(self.bertram_config.mode):
                overwrite_fct = self.get_sep_oef(ngram_vectors, nr_of_contexts)

            self.transformer.embeddings.word_embeddings.overwrite_fct = overwrite_fct
            sequence_output, _ = self.transformer(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            self.transformer.embeddings.word_embeddings.overwrite_fct = None

            # get only the mask vector position for each sequence
            # shape = sum(nr_of_contexts) x emb_dim
            mask_output = self._get_mask_output(sequence_output, mask_positions)

            # regroup the sequence_output based on given lengths
            # shape = batch_size x max(nr_of_contexts) x emb_dim where batch_size := len(nr_of_contexts)
            grouped_mask_output = Bertram._group_sequence(mask_output, nr_of_contexts)

            # shape = batch_size x max(nr_of_contexts)
            reliability_scores = self.reliability_measure(grouped_mask_output, nr_of_contexts)

            output_vectors = self._get_weighted_sum(grouped_mask_output, reliability_scores)
            output_vectors = self.linear(output_vectors)
            
            ctx_self = output_vectors
            
            
            reliability_scores[reliability_scores!=0] = 1
        
            sent_masks = reliability_scores

            old_ngram_mask = ngram_mask
        
        
            ngram_mask = ngram_mask.unsqueeze(-2).unsqueeze(-2)
            sent_masks = sent_masks.unsqueeze(-2).unsqueeze(-2)
        

            for layer in self.sub_self_att:
                ngram_embs = layer(ngram_embs, ngram_embs, mask=ngram_mask)


            
            cc_self = grouped_mask_output
            for layer in self.cc_self:
                cc_self = layer(cc_self, grouped_mask_output, mask=sent_masks)
            
            ss_self = ngram_embs
            for layer in self.ss_self:
                ss_self = layer(ss_self, ngram_embs, mask=ngram_mask)

        
            #instead of squeeze bc if ctx size is 1 or batch size is 1
            ngram_mask = torch.reshape(ngram_mask, (ngram_mask.size()[0], ngram_mask.size()[3]))
            sent_masks = torch.reshape(sent_masks, (sent_masks.size()[0], sent_masks.size()[3]))

            #squeeze() causes issue here if our ctx size is 1, so above code instead
            ngram_mask = ngram_mask.unsqueeze(-1)
            sent_masks = sent_masks.unsqueeze(-1)


            cc_mean = torch.sum(cc_self * sent_masks, dim=1) / torch.sum(sent_masks, dim=1)
            ss_mean = torch.sum(ss_self * ngram_mask, dim=1) / torch.sum(ngram_mask, dim=1)

            sub_self = torch.sum(ngram_embs * ngram_mask, dim=1) / torch.sum(ngram_mask, dim=1)




            if self.gate_combiner == 'concatenate':            
                final_out = torch.cat([ctx_self, sub_self, cc_mean, ss_mean], dim=1)
                final_out = self.final_combiner(final_out)
            elif self.gate_combiner == 'gate':

                a = torch.cat([ctx_self, sub_self, cc_mean, ss_mean], dim=1)
                alpha = self.four_combiner(a)           
                alpha = torch.softmax(alpha, dim=1) #batch * 4
                final_out = (alpha[:, 0].unsqueeze(-1) * ctx_self) + (alpha[:, 1].unsqueeze(-1) * sub_self) + (alpha[:, 2].unsqueeze(-1) * cc_mean) + (alpha[:, 3].unsqueeze(-1) * ss_mean)

            elif self.gate_combiner == 'hierarchy_gate':
                a = torch.cat([sub_self, ss_mean], dim=1)
                alpha_sub = self.two_combiner1(a)
                alpha_sub = torch.sigmoid(alpha_sub)
                final_sub_out = (alpha_sub * sub_self) + ((1-alpha_sub) * ss_mean)
        
                b = torch.cat([cc_mean, ctx_self], dim=1)
                alpha_cc = self.two_combiner2(b)
                alpha_cc = torch.sigmoid(alpha_cc)
                final_cc_out = (alpha_cc * cc_mean) + ((1-alpha_cc) * ctx_self)
        

                c = torch.cat([final_sub_out, final_cc_out], dim=1)
                alpha = self.two_combiner3(c)
                alpha = torch.sigmoid(alpha)
                final_out = (alpha * final_sub_out) + ((1- alpha) * final_cc_out)



        if target_vectors is not None:
            loss_fct = MSELoss()
            loss = loss_fct(final_out, target_vectors)
            return loss
        else:
            return final_out
            
