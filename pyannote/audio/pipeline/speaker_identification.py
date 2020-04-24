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

import numpy as np
from pathlib import Path
from typing import Optional
from typing import Union
from typing import Text
from numbers import Number

from pyannote.core import Annotation
from pyannote.database import get_annotated

from pyannote.metrics.identification import IdentificationErrorRate

from .speech_turn_segmentation import SpeechTurnSegmentation
from .speech_turn_segmentation import OracleSpeechTurnSegmentation

from pyannote.pipeline.blocks.classification import ClosestAssignment, KNN

from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform
from pyannote.audio.features.wrapper import Wrapper, Wrappable
from .utils import get_references

class SupervisedSpeakerIdentification(Pipeline):
    """Base class for Supervised Speaker identification pipelines

    Parameters
    ----------
    references : dict or Text
        Either a dict like {identity : features}
        or the name of a pyannote protocol which should be loaded like this.
    subsets: set, optional
        which protocol subset to get reference from.
        Defaults to {'train'}
    label_min_duration: float or int, optional
        Only keep speaker with at least `label_min_duration` of annotated data.
        Defaults to keep every speaker (i.e. 0.0)
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
    evaluation_only : `bool`, optional
        Only process the evaluated regions. Default to False.
    confusion : `bool`, optional
        Evaluate only confusion -> remove unknown speakers from hypothesis
        Defaults to False (i.e. evaluate with ier = confusion + false_alarm + miss)
    """

    def __init__(
        self,
        references : Union[dict, Text],
        subsets : set = {'train'},
        label_min_duration : Union[float, int] = 0.0,
        sad_scores: Union[Text, Path] = None,
        scd_scores: Union[Text, Path] = None,
        embedding: Union[Text, Path] = None,
        metric: Optional[str] = "cosine",
        evaluation_only: Optional[bool] = False,
        confusion = False
    ):

        super().__init__()
        if isinstance(references, Text):
            self.references = get_references(references,
                                             embedding,
                                             subsets,
                                             label_min_duration)
        else:
            self.references = references
        self.sad_scores = sad_scores
        self.scd_scores = scd_scores
        if self.scd_scores == "oracle":
            if self.sad_scores == "oracle":
                self.speech_turn_segmentation = OracleSpeechTurnSegmentation()
            else:
                msg = (
                    f"Both sad_scores and scd_scores should be set to 'oracle' "
                    f"for oracle speech turn segmentation, "
                    f"got {self.sad_scores} and {self.scd_scores}, respectively."
                )
                raise ValueError(msg)
        else:
            self.speech_turn_segmentation = SpeechTurnSegmentation(
                sad_scores=self.sad_scores, scd_scores=self.scd_scores
            )
        self.evaluation_only = evaluation_only
        self.confusion = confusion

        self.embedding = embedding
        self._embedding = Wrapper(self.embedding)
        self.metric = metric

    def __call__(self, current_file: dict) -> Annotation:
        """Prototype function to apply speaker identification
        Notably shows usage of evaluation_only and confusion

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.

        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Speaker identification output.
        """
        raise NotImplementedError(f'Sub-classes of {self.__class__.__name__} should '
                                  f'implement their own __call__ method')
        # segmentation into speech turns
        speech_turns = self.speech_turn_segmentation(current_file)

        # some files are only partially annotated and therefore one cannot
        # evaluate speaker diarization results on the whole file.
        # this option simply avoids trying to cluster those
        # (potentially messy) un-annotated refions by focusing only on
        # speech turns contained in the annotated regions.
        if self.evaluation_only:
            annotated = get_annotated(current_file)
            speech_turns = speech_turns.crop(annotated, mode="intersection")

        if self.confusion:
            #remove unknown speakers from hypothesis
            speech_turns = self.remove_unknown(speech_turns)
        return speech_turns

    def remove_unknown(self, hypothesis):
        result = hypothesis.empty()
        for segment, track, label in hypothesis.itertracks(yield_label=True):
            # unknown speaker (=) label < 0
            if not(isinstance(label, Number) and label < 0):
                result[segment, track] = label

        return result

    def get_metric(self) -> IdentificationErrorRate:
        """Return new instance of identification error rate metric"""
        # HACK: set miss, false_alarm to 0.1 because empty annotation gets confusion = 0.0
        # -> pipeline optimizes to get only unknown speakers (=) empty annotation
        miss, false_alarm = (0.1, 0.1) if self.confusion else (1., 1.)

        return IdentificationErrorRate(confusion=1., miss=miss, false_alarm=false_alarm,
                                       collar=0.0, skip_overlap=False)

class ClosestSpeaker(SupervisedSpeakerIdentification):
    """Speaker identification pipeline

    Parameters
    ----------
    references : dict or Text
        Either a dict like {identity : features}
        or the name of a pyannote protocol which should be loaded like this.
    subsets: set, optional
        which protocol subset to get reference from.
        Defaults to {'train'}
    label_min_duration: float or int, optional
        Only keep speaker with at least `label_min_duration` of annotated data.
        Defaults to keep every speaker (i.e. 0.0)
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
    evaluation_only : `bool`, optional
        Only process the evaluated regions. Default to False.
    confusion : `bool`, optional
        Evaluate only confusion -> remove unknown speakers from hypothesis
        Defaults to False (i.e. evaluate with ier = confusion + false_alarm + miss)
    """

    def __init__(
        self,
        references : Union[dict, Text],
        subsets : set = {'train'},
        label_min_duration : Union[float, int] = 0.0,
        sad_scores: Union[Text, Path] = None,
        scd_scores: Union[Text, Path] = None,
        embedding: Union[Text, Path] = None,
        metric: Optional[str] = "cosine",
        evaluation_only: Optional[bool] = False,
        confusion = False
    ):

        super().__init__(references, subsets, label_min_duration, sad_scores, scd_scores,
                         embedding, metric, evaluation_only, confusion)

        self.closest_assignment = ClosestAssignment(metric=self.metric)

    def __call__(self, current_file: dict, use_threshold: bool = True) -> Annotation:
        """Apply speaker identification

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.
        use_threshold : `bool`, optional
            Ignores `closest_assignment.threshold` if False
            -> sample embeddings are assigned to the closest target no matter the distance
            Defaults to True.

        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Speaker identification output.
        """
        # segmentation into speech turns
        speech_turns = self.speech_turn_segmentation(current_file)

        # some files are only partially annotated and therefore one cannot
        # evaluate speaker diarization results on the whole file.
        # this option simply avoids trying to cluster those
        # (potentially messy) un-annotated refions by focusing only on
        # speech turns contained in the annotated regions.
        if self.evaluation_only:
            annotated = get_annotated(current_file)
            speech_turns = speech_turns.crop(annotated, mode="intersection")

        # gather targets embedding
        X_targets, targets_labels = [], []
        for label, embeddings in self.references.items():
            targets_labels.append(label)
            #average embeddings per reference
            X_targets.append(np.mean(embeddings, axis=0))

        #gather inference embeddings
        embedding = self._embedding(current_file)

        # gather speech turns embedding
        labels = speech_turns.labels()
        X, assigned_labels, skipped_labels = [], [], []
        for l, label in enumerate(labels):

            timeline = speech_turns.label_timeline(label, copy=False)

            # be more and more permissive until we have
            # at least one embedding for current speech turn
            for mode in ["strict", "center", "loose"]:
                x = embedding.crop(timeline, mode=mode)
                if len(x) > 0:
                    break

            # skip labels so small we don't have any embedding for it
            if len(x) < 1:
                skipped_labels.append(label)
                continue

            assigned_labels.append(label)

            #average speech turn embeddings
            X.append(np.mean(x, axis=0))

        # assign speech turns to closest class
        assignments = self.closest_assignment(np.vstack(X_targets),
                                              np.vstack(X),
                                              use_threshold = use_threshold)
        mapping = {
            label: targets_labels[k]
            if not k < 0 else k
            for label, k in zip(assigned_labels, assignments)
        }
        speech_turns.rename_labels(mapping=mapping, copy = False)

        if self.confusion:
            #remove unknown speakers from hypothesis
            speech_turns = self.remove_unknown(speech_turns)
        return speech_turns

class KNearestSpeakers(SupervisedSpeakerIdentification):
    """Speaker identification pipeline using k-nearest neighbors

    Parameters
    ----------
    references : dict or Text
        Either a dict like {identity : features}
        or the name of a pyannote protocol which should be loaded like this.
    subsets: set, optional
        which protocol subset to get reference from.
        Defaults to {'train'}
    label_min_duration: float or int, optional
        Only keep speaker with at least `label_min_duration` of annotated data.
        Defaults to keep every speaker (i.e. 0.0)
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
    evaluation_only : `bool`, optional
        Only process the evaluated regions. Default to False.
    confusion : `bool`, optional
        Evaluate only confusion -> remove unknown speakers from hypothesis
        Defaults to False (i.e. evaluate with ier = confusion + false_alarm + miss)
    """

    def __init__(
        self,
        references : Union[dict, Text],
        subsets : set = {'train'},
        label_min_duration : Union[float, int] = 0.0,
        sad_scores: Union[Text, Path] = None,
        scd_scores: Union[Text, Path] = None,
        embedding: Union[Text, Path] = None,
        metric: Optional[str] = "cosine",
        evaluation_only: Optional[bool] = False,
        confusion = False
    ):

        super().__init__(references, subsets, label_min_duration, sad_scores, scd_scores,
                         embedding, metric, evaluation_only, confusion)
        self.classifier = KNN(self.metric)

    def __call__(self, current_file: dict, use_threshold: bool = True) -> Annotation:
        """Apply speaker identification

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.
        use_threshold : `bool`, optional
            Ignores `closest_assignment.threshold` if False
            -> sample embeddings are assigned to the closest target no matter the distance
            Defaults to True.

        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Speaker identification output.
        """

        # segmentation into speech turns
        speech_turns = self.speech_turn_segmentation(current_file)

        # some files are only partially annotated and therefore one cannot
        # evaluate speaker diarization results on the whole file.
        # this option simply avoids trying to cluster those
        # (potentially messy) un-annotated refions by focusing only on
        # speech turns contained in the annotated regions.
        if self.evaluation_only:
            annotated = get_annotated(current_file)
            speech_turns = speech_turns.crop(annotated, mode="intersection")

        # gather targets embedding
        X_targets, targets_labels = [], []
        for label, embeddings in self.references.items():
            targets_labels.extend([label for _ in embeddings])
            X_targets.extend(embeddings)

        #gather inference embeddings
        embedding = self._embedding(current_file)

        # gather speech turns embedding
        labels = speech_turns.labels()
        X, assigned_labels, skipped_labels = [], [], []
        for l, label in enumerate(labels):

            timeline = speech_turns.label_timeline(label, copy=False)

            # be more and more permissive until we have
            # at least one embedding for current speech turn
            for mode in ["strict", "center", "loose"]:
                x = embedding.crop(timeline, mode=mode)
                if len(x) > 0:
                    break

            # skip labels so small we don't have any embedding for it
            if len(x) < 1:
                skipped_labels.append(label)
                continue

            assigned_labels.append(label)

            #average speech turn embeddings
            X.append(np.mean(x, axis=0))

        # assign speech turns to the nearest neighbor
        assignments = self.classifier(np.vstack(X_targets),
                                      np.vstack(X),
                                      targets_labels,
                                      use_threshold = use_threshold)
        mapping = dict(zip(assigned_labels, assignments))
        speech_turns.rename_labels(mapping=mapping, copy = False)

        if self.confusion:
            #remove unknown speakers from hypothesis
            speech_turns = self.remove_unknown(speech_turns)
        return speech_turns
