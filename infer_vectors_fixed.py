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

"""This script can be used to infer vectors for rare words with a trained BERTRAM instance."""
import argparse
import io
import os
import re
from collections import defaultdict

import torch
import numpy as np

import log
import bertram
from input_processor import InputProcessor

from transformers import WEIGHTS_NAME
from collections import OrderedDict


logger = log.get_logger('root')


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="This script can be used to infer vectors for rare words with a"
                                                 " trained BERTRAM instance.")

    parser.add_argument('--model', '-m', type=str, required=True,
                        help="Path to the trained BERTRAM model")
    parser.add_argument('--input', '-i', type=str, required=True, nargs='+',
                        help="Path to the input files, where each line is of the form "
                             "<WORD><TAB><CONTEXT_1><TAB>...<CONTEXT_n>")
    parser.add_argument('--output', '-o', type=str, required=True,
                        help="Path were the inferred vectors are saved. Each line of the resulting file is of the form "
                             "<WORD> <VECTOR>")
    parser.add_argument('--bmode', default=None, choices=bertram.MODES,
                        help="The BERTRAM mode to use (e.g., 'add'). If not given, the trained model's standard mode "
                             "is used.")
    parser.add_argument('--max_contexts', type=int, default=100,
                        help="The maximum number of contexts per word. Words with more contexts are discarded.")
    parser.add_argument('--split_contexts', type=int, default=None,
                        help="If given, the list of contexts per word is split into chunks of size 'split_contexts'. "
                             "Each chunk is processed separately and the results are then averaged. This can be used "
                             "for words with too many contexts to fit into GPU memory.")
    parser.add_argument('--apply_internal_pca', action='store_true',
                        help="remove top components before put into rest of model, for internalPCA.")

    parser.add_argument('--skip_sub_only', action='store_true',
                        help="if sub only, skip.")



    args = parser.parse_args()

    input_processor = InputProcessor.load(os.path.join(args.model, bertram.IP_NAME))
    input_processor.ngram_dropout = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bertram_config = bertram.BertramConfig.load(os.path.join(args.model, bertram.CONFIG_NAME))
    _, _, bertram_cls = bertram.MODELS[bertram_config.transformer_cls]




    model_state_dict = torch.load(os.path.join(args.model, WEIGHTS_NAME))
    new_model_state_dict = OrderedDict()
    changed_count = 0
    for key, value in model_state_dict.items():
        #print(key)
        if 'embeddings.word_embeddings.embedding.weight' in key:
            new_key = key.replace('embeddings.word_embeddings.embedding.weight', 'embeddings.word_embeddings.weight')
            changed_count = changed_count + 1
        else:
            new_key = key
        new_model_state_dict[new_key] = value
    del model_state_dict   
    #-------------------------------- 
    
    if changed_count > 1:
        raise Exception("Should only be changing one key here")

    print(type(bertram_cls))
    
    #'''
    model, loading_info = bertram_cls.from_pretrained(args.model, bertram_config=bertram_config,
                                                   output_loading_info=True, state_dict = new_model_state_dict)  # type: bertram.Bertram

    #print('new_model_state_dict')
    #for key, value in new_model_state_dict.items():
    #    print(key)
    #print('----')
    #print('actual model')
    #for key, value in model.named_parameters():
    #    print(key)
    #print('model state_dict')
    #for key, value in model.state_dict().items():
    #    print(key)
    #print('----')
    
    
    #'''    
    if model.bertram_config.gate_combiner == 'hierarchy_gate_with_freqs':
        model.ngram_in_word_counts = input_processor.ngram_builder.ngram_in_word_counts
        model.ngram_in_word_counts[1] = 0 #hack for now    

    if model.bertram_config.gate_combiner == 'hierarchy_gate_with_freqs2' or model.bertram_config.gate_combiner == 'hierarchy_gate_with_freqs_and_sim':
        model.ngram_in_word_counts = input_processor.ngram_builder.ngram_in_word_counts
        model.ngram_in_word_counts[1] = 0 #hack for now    


    if args.bmode:
        logger.warning("Overwriting original mode {} with {}...".format(model.bertram_config.mode, args.bmode))
        model.bertram_config.mode = args.bmode
        input_processor.mode = args.bmode

    model.setup()
    

    
    model.to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    count = 0

    word2contexts = defaultdict(list)
    for inp in args.input:
        with open(inp, 'r', encoding='utf-8') as f:
            for line in f.read().splitlines():
                comps = re.split(r'\t', line)
                word = comps[0]
                contexts = [c for c in comps[1:] if c != '']
                word2contexts[word] += contexts

    with io.open(args.output, 'w', encoding='utf-8') as output_file:
        for word, contexts in word2contexts.items():
            if model.bertram_config.mode == bertram.MODE_FORM:
                contexts = [word]

            if len(contexts) >= args.max_contexts and not model.bertram_config.mode == bertram.MODE_FORM:
                logger.info('Skipping word ' + word + ' as it has %d contexts', len(contexts))
                continue

            if not contexts and model.bertram_config.mode == bertram.MODE_CONTEXT:
                logger.info('Skipping word ' + word + ' as it has no contexts')
                continue

            requires_split = args.split_contexts and contexts and len(contexts) > args.split_contexts
            contexts_set = [contexts] if not requires_split else list(chunks(contexts, args.split_contexts))
            vecs = []

            for contexts_chunk in contexts_set:
                batch = input_processor.generate_batch_from_input(word, contexts_chunk)

                if args.skip_sub_only == True:
                    if batch.input_ids is None or not contexts_chunk:
                        continue

                vec = model(
                    batch.input_ids.to(device) if contexts_chunk else None,
                    batch.segment_ids.to(device),
                    batch.nrs_of_contexts.to(device),
                    batch.mask_positions.to(device),
                    batch.input_mask.to(device),
                    batch.ngram_features.ngram_ids.to(device),
                    batch.ngram_features.ngram_lengths.to(device),
                    None
                )[0].detach().cpu().numpy()
                vecs.append(vec)

            if args.skip_sub_only == True:
                if len(vecs) == 0:
                    continue

            vec = np.mean(vecs, axis=0)

            output_file.write(word + ' ' + ' '.join([str(x) for x in vec]) + '\n')
            count += 1
            if count % 100 == 0:
                logger.info('Done processing %d words', count)
