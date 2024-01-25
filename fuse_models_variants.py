#This basically uses fuse_models to build my variants instead
#I think the main thing is "hacking" so the Config model builds the variant instead

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script fuses two BERTRAM models (form and context) to create a form-context model (ADD / REPLACE)."""
import os
import argparse
from typing import Tuple, Dict

import torch
from transformers import WEIGHTS_NAME, CONFIG_NAME

import log
import bertram
from bertram import Bertram
from input_processor import InputProcessor
from collections import OrderedDict

META_NAME = 'META_INFO.txt'
logger = log.get_logger('root')


def main():
    parser = argparse.ArgumentParser(description="This script fuses two BERTRAM models (a form and a context model)"
                                                 " to create a form-context model using either the ADD configuration"
                                                 " or the REPLACE configuration.")

    parser.add_argument('--form_model', '-form', type=str, required=True,
                        help="Path to the form-only model")
    parser.add_argument('--context_model', '-context', type=str, required=True,
                        help="Path to the context-only model")
    parser.add_argument('--mode', '-m', choices=bertram.MODES, required=True,
                        help="Mode for the resulting model (e.g. 'add')")
    parser.add_argument('--output', '-o', type=str, required=True,
                        help="Path to save the combined model")
    parser.add_argument('--variant_model_type', '-var', type=str, required=True,
                        help="variant model type")
    parser.add_argument('--transformer_dropout', '-trans_drop', default=0.1, type=float, required=False,
                        help="transformer_dropout")
    parser.add_argument('--variant_num_layers', '-numlyr', default=1, type=int, required=False,
                        help="variant_num_layers")
    parser.add_argument('--variant_sub_num_layers', '-numsublyr', default=1, type=int, required=False,
                        help="variant_sub_num_layers")
    parser.add_argument('--gate_combiner', '-gc', default='concatenate', type=str, required=False,
                        help="gate_combiner")


    args = parser.parse_args()

    assert bertram.requires_form(args.mode) and bertram.requires_context(args.mode)

    if os.path.exists(args.output) and os.listdir(args.output):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output))
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    ip_form, bc_form, model_form = _load_model_components(args.form_model)
    ip_context, bc_context, model_context = _load_model_components(args.context_model)

    input_processor = _fuse_input_processors(ip_form, ip_context)
    bertram_config = _fuse_bertram_configs(bc_form, bc_context, args.mode, args.variant_model_type, transformer_dropout=args.transformer_dropout, num_layers=args.variant_num_layers, sub_num_layers=args.variant_sub_num_layers, gate_combiner=args.gate_combiner)
    model = _fuse_models(model_form, model_context, bertram_config, args.variant_model_type, bc_context)  # type: Bertram

    model.bertram_config.mode = args.mode
    meta_info = {'form_model': args.form_model, 'context_model': args.context_model}

    _save_model_components(args.output, input_processor, bertram_config, model, meta_info)


def _fuse_input_processors(ip_form: InputProcessor, ip_context: InputProcessor) -> InputProcessor:
    assert ip_form.word_embeddings_file == ip_context.word_embeddings_file
    assert ip_form.word_embeddings_format == ip_context.word_embeddings_format
    assert set(ip_form.train_files) == set(ip_context.train_files)
    #assert ip_form.vocab_file == ip_context.vocab_file
    assert ip_form.vector_size == ip_context.vector_size
    assert ip_form.min_word_count == ip_context.min_word_count
    assert ip_form.sep_symbol == ip_context.sep_symbol

    return InputProcessor(
        word_embeddings_file=ip_form.word_embeddings_file,
        word_embeddings_format=ip_form.word_embeddings_format,
        train_files=ip_form.train_files,
        vocab_file=ip_form.vocab_file,
        vector_size=ip_form.vector_size,
        ngram_threshold=ip_form.ngram_threshold,
        nmin=ip_form.nmin,
        nmax=ip_form.nmax,
        ngram_dropout=ip_form.ngram_dropout,
        min_word_count=ip_form.min_word_count,
        max_copies=ip_context.max_copies,
        smin=ip_context.smin,
        smax=ip_context.smax,
        max_seq_length=ip_context.max_seq_length,
        model_cls=ip_context.model_cls,
        bert_model=ip_context.bert_model
    )


def _fuse_bertram_configs(bc_form: bertram.BertramConfig, bc_context: bertram.BertramConfig,
                          mode: str, variant_model_type: str, transformer_dropout, num_layers, sub_num_layers, gate_combiner) -> bertram.BertramConfig:
    assert bc_form.output_size == bc_context.output_size
    
    #so hear, we need to change the transformer_cls to our desired model
    
    return bertram.BertramConfig(mode=mode, output_size=bc_form.output_size,
                                 ngram_vocab_size=bc_form.ngram_vocab_size, transformer_cls=variant_model_type, transformer_dropout=transformer_dropout, variant_num_layers=num_layers, variant_sub_num_layers=sub_num_layers, gate_combiner=gate_combiner)


def _fuse_models(model_form: bertram.Bertram, model_context: bertram.Bertram,
                 bertram_config: bertram.BertramConfig, variant_model_type: str, bc_context: bertram.BertramConfig) -> bertram.Bertram:
    _, _, bertram_cls = bertram.MODELS[bertram_config.transformer_cls]
    
    #model_context.config, is this correct?  YES.  This is transformer config
    model = bertram_cls(model_context.config, bertram_config)
    
    #print(model.state_dict())
    #print('--')
    #print(model_context.state_dict())
    
    #need to keyname keys
    new_model_context_state_dict = OrderedDict()
    old_dict = model_context.state_dict()
    for key, value in old_dict.items():
        #print(key, value)
        #print(key)
        new_key = key.replace(bc_context.transformer_cls, str(variant_model_type))
        #print(new_key)
        new_model_context_state_dict[new_key] = value
        #print('---')


    
    #'''
    #print('--------')        
    #for key, value in new_model_context_state_dict.items():
    #    print(key)
    #print('--------')        
    #for key, value in model.state_dict().items():
    #    print(key)
    
    #print('cat'-2)
    #'''
    
    
    model.load_state_dict(new_model_context_state_dict, strict=False)
    model.ngram_processor.load_state_dict(model_form.ngram_processor.state_dict(), strict=False)
    
    #for name, param in model.named_parameters():
    #    print(str(name)+" : "+str(param.data))
        
    #print('cat'-2)
    
    return model


def _load_model_components(path: str) -> Tuple[InputProcessor, bertram.BertramConfig, bertram.Bertram]:
    input_processor = InputProcessor.load(os.path.join(path, bertram.IP_NAME))
    bertram_config = bertram.BertramConfig.load(os.path.join(path, bertram.CONFIG_NAME))
    _, _, bertram_cls = bertram.MODELS[bertram_config.transformer_cls]
    model = bertram_cls.from_pretrained(path, bertram_config=bertram_config)
    return input_processor, bertram_config, model


def _save_model_components(path: str, input_processor: InputProcessor, bertram_config: bertram.BertramConfig,
                           model: bertram.Bertram, meta_info: Dict) -> None:
    output_ip_file = os.path.join(path, bertram.IP_NAME)
    input_processor.save(output_ip_file)

    output_bc_file = os.path.join(path, bertram.CONFIG_NAME)
    bertram_config.save(output_bc_file)

    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(path, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    output_config_file = os.path.join(path, CONFIG_NAME)
    with open(output_config_file, 'w') as f:
        f.write(model_to_save.config.to_json_string())

    meta_file = os.path.join(path, META_NAME)
    with open(meta_file, 'w') as f:
        for k, v in meta_info.items():
            f.write('{}: {}\n'.format(k, v))
            
    param_out_file = os.path.join(path, 'param_summary.txt')
    param_file = open(param_out_file, 'w')
    for name, param in model.named_parameters():
        param_file.write(str(name)+" : "+str(param.data)+'\n')
    param_file.close


if __name__ == '__main__':
    main()
