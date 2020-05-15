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
from typing import Optional
from typing import Union
from typing import Text
import numpy as np

from pyannote.core import Annotation
from pyannote.database import get_annotated

from .speaker_identification import SpeakerIdentification
from .speaker_diarization import SpeakerDiarization
from .ASR import OracleASR, OracleNormalizer, PLAIN, NORMALIZED

class LateFusion(SpeakerIdentification):
    """Base class for late fusion for named speaker identification

    Fuses :
    - diarization pipeline
    - normalized ASR pipeline

    Parameters
    ----------
    sad_scores : Text or Path or 'oracle', optional
        Describes how raw speech activity detection scores
        should be obtained. It can be either the name of a torch.hub model, or
        the path to the output of the validation step of a model trained
        locally, or the path to scores precomputed on disk.
        Defaults to "@sad_scores", indicating that protocol
        files provide the scores in the corresponding "sad_scores" key.
        Use 'oracle' to assume perfect speech activity detection.
    scd_scores : Text or Path or 'oracle', optional
        Describes how raw speaker change detection scores
        should be obtained. It can be either the name of a torch.hub model, or
        the path to the output of the validation step of a model trained
        locally, or the path to scores precomputed on disk.
        Defaults to "@scd_scores", indicating that protocol
        files provide the scores in the corresponding "scd_scores" key.
        Use 'oracle' to assume perfect speech turn segmentation,
        `sad_scores` should then be set to 'oracle' too.
    embedding : Text or Path, optional
        Describes how raw speaker embeddings should be obtained. It can be
        either the name of a torch.hub model, or the path to the output of the
        validation step of a model trained locally, or the path to embeddings
        precomputed on disk. Defaults to "@emb" that indicates that protocol
        files provide the embeddings in the "emb" key.
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Metric used for comparing embeddings. Defaults to 'cosine'.
    method : {'pool', 'affinity_propagation'}
        Clustering method. Defaults to 'pool'.
    evaluation_only : `bool`
        Only process the evaluated regions. Default to False.
    purity : `float`, optional
        Optimize coverage for target purity.
        Defaults to optimizing diarization error rate.
    """

    def __init__(
        self,
        sad_scores: Union[Text, Path] = None,
        scd_scores: Union[Text, Path] = None,
        embedding: Union[Text, Path] = None,
        metric: Optional[str] = "cosine",
        method: Optional[str] = "pool",
        evaluation_only: Optional[bool] = False,
        purity: Optional[float] = None
    ):

        super().__init__()
        self.diarization = SpeakerDiarization(sad_scores,
                                              scd_scores,
                                              embedding,
                                              metric,
                                              method,
                                              evaluation_only,
                                              purity)
        self.ASR = OracleNormalizer()

    def keep_normalized(self, annotation):
        """return annotation with only normalized names"""
        normalized = annotation.empty()
        for segment, track, name in annotation.itertracks(yield_label=True):
            if track == NORMALIZED:
                normalized[segment, track] = name

        return normalized

class NaiveMapping(LateFusion):
    """Naive Mapping for named speaker identification
    Assigns to each cluster the closest (in time) name mention
    """

    def __call__(self, current_file: dict) -> Annotation:
        """Apply named speaker identification

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.

        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Speaker identification output.
        """

        # diarization output
        clusters = self.diarization(current_file)

        # normalized ASR output
        text = self.ASR(current_file)

        # keep only normalized names
        text = self.keep_normalized(text)

        # map clusters to name in text.labels()
        mapping = {}
        for cluster in clusters.labels():
            # 1. compute duration between cluster segments and names
            scores = {}
            timeline = clusters.label_timeline(cluster, copy=False)
            for segment in timeline:
                for name in text.labels():
                    scores.setdefault(name, 0.0)
                    name_timeline = text.label_timeline(name, copy=False)
                    for name_segment in name_timeline:
                        distance = abs(segment.middle - name_segment.middle)
                        scores[name] += distance / len(name_timeline)
            # 2. keep the closest name
            mapping[cluster] = min(scores, key=scores.get)

        # 3. do the actual mapping
        return clusters.rename_labels(mapping=mapping)


