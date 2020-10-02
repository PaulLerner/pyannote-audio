#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2020 CNRS

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
from typing import Union
from typing import Text

from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment

from pyannote.core import Annotation
from pyannote.metrics.identification import IdentificationErrorRate
from pyannote.pipeline import Pipeline
from pyannote.database.util import load_id
from .speaker_diarization import SpeakerDiarization


class SpeakerIdentification(Pipeline):
    """Base class for speaker identification pipeline

    Defines metric as IdentificationErrorRate
    """
    def get_metric(self) -> IdentificationErrorRate:
        """Return new instance of diarization error rate metric"""
        return IdentificationErrorRate(collar=0.0, skip_overlap=False)


class LateFusion(SpeakerIdentification):
    """Base class for late-fusion speaker identification pipeline

    Takes as input:
    - Identification hypothesis
    - Diarization hypothesis

    and merges the two

    Parameters
    ----------
    identification: Text or Path
        Path towards the identification hypothesis in RTTM format
    **kwargs: Additional parameters are passed to the diarization pipeline
    """

    def __init__(self, identification: Union[Text, Path], **kwargs):
        super().__init__()
        # load identification hypothesis from id (RTTM-like)
        self.identification = load_id(identification)

        # init diarization pipeline
        self.diarization = SpeakerDiarization(**kwargs)


def confidence_cooccurrence(annotation, identification):
    """Same as annotation*other but segment duration doesn't weigh in.
    Instead, the co-occurence score comes from identification confidences scores (sadly stored as "track")"""
    # TODO: move to pyannote.core if appropriate
    i_labels = annotation.labels()
    j_labels = identification.labels()

    I = {label: i for i, label in enumerate(i_labels)}
    J = {label: j for j, label in enumerate(j_labels)}

    matrix = np.zeros((len(I), len(J)))

    # iterate over intersecting tracks and accumulate co-occurrence
    for (segment, track), (id_segment, id_conf) in annotation.co_iter(identification):
        i = I[annotation[segment, track]]
        j = J[identification[id_segment, id_conf]]
        matrix[i, j] += id_conf

    return matrix


class MajorityVoting(LateFusion):
    """Fuses identification and diarization hypotheses using majority voting
    The identity of each cluster is the mode of the identification hypothesis

    See LateFusion
    """

    def __call__(self, current_file: dict) -> Annotation:
        """Apply majority voting to fuse speaker diarization and identification

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.

        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Speaker identification output.
        """
        uri = current_file["uri"]
        identification = self.identification[uri]
        diarization = self.diarization(current_file)

        # find one-to-one mapping that maximes co-occurence between
        # diarization and identification using the Hungarian algorithm
        mapping = {}
        cooccurrence = confidence_cooccurrence(diarization, identification)
        d_labels, i_labels = diarization.labels(), identification.labels()

        for a, b in zip(*linear_sum_assignment(-cooccurrence)):
            if cooccurrence[a, b] > 0:
                mapping[d_labels[a]] = i_labels[b]

        # update diarization hypothesis with speaker id
        return diarization.rename_labels(mapping=mapping)