import random
import re
import os
import datasets


model = 'r1-llama'
exp = 2

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        if exp == 1:  # non-en
            if model == 'qwen':
                # qwen: ar,cs,es,ko
                languages = ["Arabic", "Czech", "Spanish", "Korean"]
            elif model == 'llama':
                # llama: ar,ru,sr,vi
                languages = ["Arabic", "Russian", "Serbian", "Vietnamese"]
            elif model == 'r1-llama':
                # r1-llama: ar,bn,de,sr
                languages = ["Arabic", "Bengali", "German", "Serbian"]
        elif exp == 2:  # with en
            if model == 'qwen':
                # qwen: ar,cs,en,ko
                languages = ["Arabic", "Czech", "English", "Korean"]
            elif model == 'llama':
                # llama: ar,ru,en,vi
                languages = ["Arabic", "Russian", "English", "Vietnamese"]
            elif model == 'r1-llama':
                # r1-llama: ar,en,de,sr
                languages = ["Arabic", "English", "German", "Serbian"]
        
        random.shuffle(languages)

        for i_lang in range(4):
            doc[f"lang{i_lang+1}"] = languages[i_lang]
            
        return doc

    return dataset.map(_process_doc)
