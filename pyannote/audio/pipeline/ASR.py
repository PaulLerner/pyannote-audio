#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Paul LERNER

from pathlib import Path
from string import punctuation

from pyannote.core import Annotation, Segment

from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform


# track names to separate normalized names from plain text
PLAIN = "plain"
NORMALIZED = "normalized"
# normalized names we don't want to consider as annotated
NA = {'UNKNOWN', 'multiple_persons'}

def parse_aligned(annotations, uri):
    """
    Parameters
    ----------
    annotations: `str` or `Path`
        path to the annotations.
    uri : `str
        uri of the file

    Returns
    -------
    annotation: List[Tuple[Segment, str, float]]]
    """
    annotation = []
    aligned = annotations / f'{uri}.aligned'

    with open(aligned) as file:
        aligned = file.read().split('\n')

    for line in aligned:
        if line == '':
            continue
        _, _, start, end, text, confidence = line.split()
        start, end, confidence = map(float, (start, end, confidence))
        segment = Segment(start, end)
        # if text == "…":
        #     text = "..."
        annotation.append((segment, text, confidence))

    return annotation

class OracleASR(Pipeline):
    """
    Parameters
    ----------
    asr: `str` or `Path`
        path to the ASR annotations.
        only the '.aligned' format described in pyannote.db.plumcot
        is supported for now

    Hyperparameters
    ---------------
    confidence: `float`
        only keep words with a confidence above `threshold`
    """
    def __init__(self, annotations, asr):
        super().__init__()
        self.annotations = Path(annotations)
        self.asr = Path(asr)
        self.confidence = Uniform(0, 1)

    def __call__(self, current_file: dict) -> Annotation:
        """Parses annotations given file uri in an Annotation"""
        uri = current_file['uri']
        plain_text = parse_aligned(self.asr, uri)
        annotation = Annotation(uri, modality='text')

        for segment, text, confidence in plain_text:
            if confidence < self.confidence:
                continue
            annotation[segment, PLAIN] = text

        return annotation

def strip_punct(string):
    return string.translate(str.maketrans('', '', punctuation))

def strip_dots(string):
    return string.replace('.','')

class OracleNormalizer(Pipeline):
    """
    Parameters
    ----------
    annotations: `str` or `Path`
        path to the annotations.
        only the '.csv' format described in pyannote.db.plumcot
        is supported for now
    asr: `str` or `Path`
        path to the ASR annotations.
        only the '.aligned' format described in pyannote.db.plumcot
        is supported for now

    Hyperparameters
    ---------------
    confidence: `float`
        only keep words with a confidence above `threshold`
    """

    def __init__(self, annotations, asr):
        super().__init__()
        self.annotations = Path(annotations)
        self.asr = Path(asr)
        self.confidence = Uniform(0, 1)

    def __call__(self, current_file: dict) -> Annotation:
        """Parses annotations given file uri in an Annotation"""
        uri = current_file['uri']
        plain_text = parse_aligned(self.asr, uri)
        annotation = Annotation(uri, modality='text')

        # 1. read entities csv
        entities = self.annotations / f'merge_{uri}.csv'
        with open(entities) as file:
            entities = file.read().split('\n')

        # 2. parse entities csv to list
        normalized = []
        for line in entities[1:]:
            if line == '':
                continue
            _, _, token, _, pos, _, _, _, _, _, _, _, _, _, _, label = line.split(';')
            token, pos, label = map(str.strip, (token, pos, label))
            # remove empty lines
            if token == '':
                continue
            # first token of each line includes speaker names
            token = token[token.find(' ') + 1:]

            normalized.append((token, pos, label))

        # 3. merge entities with ASR
        i = 0
        for segment, text, confidence in plain_text:
            skip_text, tokenization = False, False
            token, pos, label = normalized[i]
            # handle tokenization
            while token != text and i+1 < len(normalized):
                # handle weird '"' corner-case
                if '"' in token:
                    break
                # handle punctuation
                elif token== '.':
                    i+=1
                    token, pos, label = normalized[i]
                elif strip_dots(token) == strip_dots(text):
                    break
                elif text == '.':
                    skip_text = True
                    break
                # token was split by a tokenizer at some point
                else:
                    t, p, l = normalized[i+1]
                    token += t
                    pos += p
                    label += l
                    i += 1
            if skip_text:
                continue
            i += 1

            if confidence < self.confidence:
                continue

            annotation[segment, PLAIN] = text

            # keep only proper names
            if label != '' and pos == 'PROPN' and label not in NA:
                annotation[segment, NORMALIZED] = label

        return annotation
