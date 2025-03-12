import random
import re

import datasets


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

        # languages = ["Spanish", "Japanese", "Korean", "Swahili"]
        languages = ["Spanish", "Japanese", "Korean", "English"]
        
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
