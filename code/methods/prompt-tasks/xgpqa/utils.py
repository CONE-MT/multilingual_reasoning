import random
import re
import os
import datasets


model = 'r1-llama'
exp = 2

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        if exp == 1:  # non-en
            if model == 'qwen':
                # qwen: fr,es,ja,th
                languages = ["French", "Spanish", "Japanese", "Thai"]
            elif model == 'llama':
                # llama: fr,ko,sw,vi
                languages = ["French", "Korean", "Swahili", "Vietnamese"]
            elif model == 'r1-llama':
                # r1-llama: ar,es,ko,sr
                languages = ["Arabic", "Spanish", "Korean", "Serbian"]
        elif exp == 2:  # with en
            if model == 'qwen':
                # qwen: en,es,ja,th
                languages = ["English", "Spanish", "Japanese", "Thai"]
            elif model == 'llama':
                # llama: fr,ko,en,vi
                languages = ["French", "Korean", "English", "Vietnamese"]
            elif model == 'r1-llama':
                # r1-llama: ar,es,ko,en
                languages = ["Arabic", "Spanish", "Korean", "English"]
        
        random.shuffle(choices)
        random.shuffle(languages)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "lang1": languages[0],
            "lang2": languages[1],
            "lang3": languages[2],
            "lang4": languages[3],
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "choices": [choices[0], choices[1], choices[2], choices[3]],
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)
