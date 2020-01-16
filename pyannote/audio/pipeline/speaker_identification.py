#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2019 CNRS

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
# Hervé BREDIN - http://herve.niderb.fr
# Paul LERNER

from pathlib import Path

from pyannote.core import Annotation
from pyannote.database import get_annotated

from pyannote.metrics.identification import IdentificationErrorRate

from .speech_turn_segmentation import SpeechTurnSegmentation
from .speech_turn_clustering import SpeechTurnClustering
from .speech_turn_assignment import SpeechTurnDatabaseAssignment

from typing import Optional
from typing import Union
from pyannote.pipeline import Pipeline
from pyannote.pipeline import SpeakerDiarization
from pyannote.pipeline.parameter import Uniform

class SpeakerIdentification(Pipeline):
    """Speaker identification pipeline

    Parameters
    ----------
    protocol_name : `str`
        Name of speaker verification protocol
    sad_scores : `Path` or 'oracle'
        Path to precomputed speech activity detection scores.
        Use 'oracle' to assume perfect speech activity detection.
    scd_scores : `Path` or 'oracle'
        Path to precomputed SCD scores on disk.
        Use 'oracle' to assume perfect speaker change detection.
    embedding : `Path`
        Path to precomputed embedding on disk
    method : {'pool', 'affinity_propagation'}, optional
        Clustering method. Defaults to 'pool'.
        If None is provided (default), the identification will be done
        directly over the speech turn segmentation.
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Metric used for comparing embeddings. Defaults to 'cosine'.
    evaluation_only : `bool`
        Only process the evaluated regions. Default to False.
    credits : `Path`, optional
        Path to the `database` credits.
        If provided, the model will only assign labels that are in
        credits[current_file['uri']].
        Defaults to None (i.e. the model can assign any label in the database)
    """

    def __init__(self, protocol_name: str,
                       sad_scores: Optional[Union[Path, str]] = None,
                       scd_scores: Optional[Union[Path, str]] = None,
                       embedding: Optional[Path] = None,
                       metric: Optional[str] = 'cosine',
                       method: Optional[str] = None,
                       evaluation_only: Optional[bool] = False
                       credits: Optional[Path] = None):

        super().__init__()
        self.protocol_name = protocol_name
        self.sad_scores = sad_scores
        self.scd_scores = scd_scores
        self.speech_turn_segmentation = SpeechTurnSegmentation(
            sad_scores=self.sad_scores,
            scd_scores=self.scd_scores)
        self.evaluation_only = evaluation_only

        self.embedding = embedding
        self.metric = metric
        if method:
            self.speaker_diarization = SpeakerDiarization(self.sad_scores,
                self.scd_scores, self.embedding, self.metric,
                method, self.evaluation_only)
        else:
            self.speaker_diarization=None

        self.speech_turn_assignment = SpeechTurnDatabaseAssignment(self.protocol_name,
            self.embedding, self.metric, credits)

    def __call__(self, current_file: dict, subset: Optional[str] = 'train') -> Annotation:
        """Apply speaker identification

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.
        subset : {'train', 'development', 'test'}, optional
            Name of subset. Defaults to 'train'
        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Speaker identification output.
        """
        if self.speaker_diarization:
            speech_turns=self.speaker_diarization(current_file)
        else:
            # segmentation into speech turns
            speech_turns = self.speech_turn_segmentation(current_file)

            # some files are only partially annotated and therefore one cannot
            # evaluate speaker identification results on the whole file.
            # this option simply avoids trying to cluster those
            # (potentially messy) un-annotated refions by focusing only on
            # speech turns contained in the annotated regions.
            if self.evaluation_only:
                annotated = get_annotated(current_file)
                speech_turns = speech_turns.crop(annotated, mode='intersection')
            speech_turns.rename_labels(generator='int', copy=False)

        speech_turns  = self.speech_turn_assignment(current_file,
                                                   speech_turns,
                                                   subset)
        return speech_turns, distances

    def get_metric(self) -> IdentificationErrorRate:
        """Return new instance of identification error rate metric"""

        return IdentificationErrorRate(collar=0.0, skip_overlap=False)
