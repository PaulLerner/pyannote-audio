#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2018 CNRS

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


from typing import Optional
from pathlib import Path
import numpy as np

from pyannote.database import get_protocol
from pyannote.pipeline import Pipeline
from pyannote.pipeline.blocks.classification import ClosestAssignment
from pyannote.core import Annotation
from .utils import assert_int_labels
from .utils import assert_string_labels
from ..features import Precomputed

import pyannote.database
from Plumcot import Plumcot



class SpeechTurnDatabaseAssignment(Pipeline):
    """Assign speech turn to closest target in the whole database

    Parameters
    ----------
    protocol_name : `str`
        Name of speaker verification protocol
    embedding : `Path`
        Path to precomputed embeddings.
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Metric used for comparing embeddings. Defaults to 'cosine'.
    serie_uri : `str`, optional
        Uri of the Plumcot serie.
        If provided, the model will only assign labels that are in the relevant file.
        Defaults to None (i.e. the model can assign any label in the database)
    """

    def __init__(self, protocol_name: str,
                       embedding: Optional[Path] = None,
                       metric: Optional[str] = 'cosine',
                       serie_uri: Optional[str] = None):
        super().__init__()

        self.protocol = get_protocol(protocol_name)
        self.embedding = embedding
        self.precomputed_ = Precomputed(self.embedding)

        self.metric = metric

        self.closest_assignment = ClosestAssignment(metric=self.metric)
        self.serie_uri=serie_uri
        if self.serie_uri:
            db=Plumcot()
            self.characters=db.get_characters(self.serie_uri)

    def __call__(self, current_file: dict,
                       speech_turns: Annotation,
                       subset: Optional[str] = 'train') -> Annotation:
        """Assign each speech turn to closest target (if close enough)

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.
        speech_turns : `Annotation`
            Speech turns. Should only contain `int` labels.
        subset : {'train', 'development', 'test'}, optional
            Name of subset. Defaults to 'train'

        Returns
        -------
        assigned : `Annotation`
            Assigned speech turns.
        """


        assert_int_labels(speech_turns, 'speech_turns')

        targets_dict = {}
        #gather all target embeddings on the whole data subset


        for target_file in getattr(self.protocol, subset)():
            target_embedding = self.precomputed_(target_file)
            annotated=target_file['annotated']
            targets=target_file['annotation'].crop(annotated,mode='intersection')
            assert_string_labels(targets, 'targets')
            # gather targets embedding
            labels = targets.labels()
            for l, label in enumerate(labels):
                if self.characters:#assist model : remove irrelevant targets
                    if label not in self.characters[current_file['uri']]:
                        continue

                timeline = targets.label_timeline(label, copy=False)

                # be more and more permissive until we have
                # at least one embedding for current speech turn
                for mode in ['strict', 'center', 'loose']:
                    x = target_embedding.crop(timeline, mode=mode)
                    if len(x) > 0:
                        break

                # skip labels so small we don't have any embedding for it
                if len(x) < 1:
                    continue
                if label in targets_dict:
                    targets_dict[label].append(x)
                else:
                    targets_dict[label]=[x]

        X_targets, targets_labels = [],[]
        #average embedding per target
        embedding = self.precomputed_(current_file)
        for label, x in targets_dict.items():
            targets_labels.append(label)
            x=np.concatenate(x, axis=0)
            X_targets.append(np.mean(x, axis=0))

        # gather speech turns embedding
        labels = speech_turns.labels()
        X, assigned_labels, skipped_labels, timelines = [], [], [], []
        for l, label in enumerate(labels):

            timeline = speech_turns.label_timeline(label, copy=False)

            # be more and more permissive until we have
            # at least one embedding for current speech turn
            for mode in ['strict', 'center', 'loose']:
                x = embedding.crop(timeline, mode=mode)
                if len(x) > 0:
                    break

            # skip labels so small we don't have any embedding for it
            if len(x) < 1:
                skipped_labels.append(label)
                continue

            assigned_labels.append(label)
            timelines.append(timeline)
            X.append(np.mean(x, axis=0))

        # assign speech turns to closest class
        assignments, distances = self.closest_assignment(np.vstack(X_targets),
                                                         np.vstack(X))
        mapping = {label: targets_labels[k]
                   for label, k in zip(assigned_labels, assignments)
                   if not k < 0}
        return speech_turns.rename_labels(mapping=mapping), distances, timelines

class SpeechTurnClosestAssignment(Pipeline):
    """Assign speech turn to closest cluster

    Parameters
    ----------
    embedding : `Path`
        Path to precomputed embeddings.
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Metric used for comparing embeddings. Defaults to 'cosine'.
    """

    def __init__(self, embedding: Optional[Path] = None,
                       metric: Optional[str] = 'cosine'):
        super().__init__()

        self.embedding = embedding
        self.precomputed_ = Precomputed(self.embedding)

        self.metric = metric

        self.closest_assignment = ClosestAssignment(metric=self.metric)

    def __call__(self, current_file: dict,
                       speech_turns: Annotation,
                       targets: Annotation) -> Annotation:
        """Assign each speech turn to closest target (if close enough)

        Parameters
        ----------
        current_file : `dict`
            File as provided by a pyannote.database protocol.
        speech_turns : `Annotation`
            Speech turns. Should only contain `int` labels.
        targets : `Annotation`
            Targets. Should only contain `str` labels.

        Returns
        -------
        assigned : `Annotation`
            Assigned speech turns.
        """

        assert_string_labels(targets, 'targets')
        assert_int_labels(speech_turns, 'speech_turns')

        embedding = self.precomputed_(current_file)

        # gather targets embedding
        labels = targets.labels()
        X_targets, targets_labels = [], []
        for l, label in enumerate(labels):

            timeline = targets.label_timeline(label, copy=False)

            # be more and more permissive until we have
            # at least one embedding for current speech turn
            for mode in ['strict', 'center', 'loose']:
                x = embedding.crop(timeline, mode=mode)
                if len(x) > 0:
                    break

            # skip labels so small we don't have any embedding for it
            if len(x) < 1:
                continue

            targets_labels.append(label)
            X_targets.append(np.mean(x, axis=0))

        # gather speech turns embedding
        labels = speech_turns.labels()
        X, assigned_labels, skipped_labels = [], [], []
        for l, label in enumerate(labels):

            timeline = speech_turns.label_timeline(label, copy=False)

            # be more and more permissive until we have
            # at least one embedding for current speech turn
            for mode in ['strict', 'center', 'loose']:
                x = embedding.crop(timeline, mode=mode)
                if len(x) > 0:
                    break

            # skip labels so small we don't have any embedding for it
            if len(x) < 1:
                skipped_labels.append(label)
                continue

            assigned_labels.append(label)
            X.append(np.mean(x, axis=0))

        # assign speech turns to closest class
        assignments, _ = self.closest_assignment(np.vstack(X_targets),
                                              np.vstack(X))
        mapping = {label: targets_labels[k]
                   for label, k in zip(assigned_labels, assignments)
                   if not k < 0}
        return speech_turns.rename_labels(mapping=mapping)
