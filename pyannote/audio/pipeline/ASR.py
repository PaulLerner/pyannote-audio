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

from pyannote.core import Annotation, Segment

from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform


# track names to separate normalized names from plain text
PLAIN = "plain"
NORMALIZED = "normalized"
# normalized names we don't want to consider as annotated
NA = {'UNKNOWN', 'multiple_persons'}

class OracleASR(Pipeline):
    """Timestamped transcriptions.

    Hyperparameters
    ---------------
    confidence: `float`
        only keep words with a confidence above `threshold`
    """
    def __init__(self):
        super().__init__()
        self.confidence = Uniform(0, 1)

    def __call__(self, current_file: dict) -> Annotation:
        """Convert spaCy Doc to Annotation"""
        uri = current_file['uri']
        annotation = Annotation(uri, modality='text')

        transcription = current_file['transcription']
        for token in transcription:
            if token._.confidence < self.confidence:
                continue
            segment = Segment(token._.time_start, token._.time_end)
            annotation[segment, PLAIN] = token.text

        return annotation

class OracleNormalizer(Pipeline):
    """Timestamped transcriptions with normalized (proper) names on a separate track.

    Hyperparameters
    ---------------
    confidence: `float`
        only keep words with a confidence above `threshold`
    """

    def __init__(self):
        super().__init__()
        self.confidence = Uniform(0, 1)

    def __call__(self, current_file: dict) -> Annotation:
        """Convert spaCy Doc to Annotation.
        Only proper names are kept (i.e. pronouns are discarded).
        Also, ambiguous labels such as 'UNKNOWN', 'multiple_persons' are discarded.
        """
        uri = current_file['uri']
        annotation = Annotation(uri, modality='text')

        entity = current_file['entity']
        for token in entity:
            if token._.confidence < self.confidence:
                continue
            segment = Segment(token._.time_start, token._.time_end)
            annotation[segment, PLAIN] = token.text
            label = token.ent_kb_id_
            # keep only proper names
            if label != '' and token.pos_ == 'PROPN' and label not in NA:
                annotation[segment, NORMALIZED] = label

        return annotation
