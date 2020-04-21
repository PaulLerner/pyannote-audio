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
import yaml
from pathlib import Path
from pyannote.core import Annotation
from pyannote.pipeline import Pipeline
from pyannote.core.utils.helper import get_class_by_name
from pyannote.database import get_protocol
from pyannote.audio.features.wrapper import Wrapper, Wrappable
from pyannote.database import FileFinder

def assert_string_labels(annotation: Annotation, name: str):
    """Check that annotation only contains string labels

    Parameters
    ----------
    annotation : `pyannote.core.Annotation`
        Annotation.
    name : `str`
        Name of the annotation (used for user feedback in case of failure)
    """
    if any(not isinstance(label, str) for label in annotation.labels()):
        msg = f"{name} must contain `str` labels only."
        raise ValueError(msg)


def assert_int_labels(annotation: Annotation, name: str):
    """Check that annotation only contains integer labels

    Parameters
    ----------
    annotation : `pyannote.core.Annotation`
        Annotation.
    name : `str`
        Name of the annotation (used for user feedback in case of failure)
    """
    if any(not isinstance(label, int) for label in annotation.labels()):
        msg = f"{name} must contain `int` labels only."
        raise ValueError(msg)

def get_references(protocol: str,
                   model: Wrappable = "@emb",
                   subsets: set = {'train'},
                   label_min_duration: Union[float, int] = 0.0):
    """Gets references from protocol
    Parameters
    ----------
    protocol: str
    model: Wrappable, optional
        Describes how raw speaker embeddings should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.
        Defaults to "@emb" that indicates that protocol files provide
        the scores in the "emb" key.
    subsets: set, optional
        which protocol subset to get reference from.
        Defaults to {'train'}
    label_min_duration: float or int, optional
        Only keep speaker with at least `label_min_duration` of annotated data.
        Defaults to keep every speaker (i.e. 0.0)

    Returns
    -------
    references : dict
        a dict like {identity : embeddings}
        with embeddings being a list of embeddings
    """
    references, durations = {}, {}
    preprocessors = {'audio': FileFinder()}
    protocol = get_protocol(protocol, preprocessors=preprocessors)
    model = Wrapper(model)
    for subset in subsets:
        for current_file in getattr(protocol, subset)():
            embedding = model(current_file)
            annotation = current_file['annotation']
            labels = annotation.labels()
            for l, label in enumerate(labels):
                timeline = annotation.label_timeline(label, copy=False)

                # be more and more permissive until we have
                # at least one embedding for current speech turn
                for mode in ['strict', 'center', 'loose']:
                    x = embedding.crop(timeline, mode=mode)
                    if len(x) > 0:
                        break

                # skip labels so small we don't have any embedding for it
                if len(x) < 1:
                    continue

                #average label embeddings
                x = np.mean(x, axis=0)

                #append reference to the references
                references.setdefault(label,[])
                references[label].append(x)

                #keep track of label duration
                durations.setdefault(label,0.)
                durations[label]+=timeline.duration()

    #filter out labels based on label_min_duration
    references = {speaker:embeddings for speaker,embeddings in references.items()
                                     if durations[speaker] > label_min_duration}
    return references

def update_references(current_file: dict,
                      annotation: Annotation,
                      model: Wrappable = "@emb",
                      references = {}):
    """Updates references from annotation"""
    annotation = current_file['annotation']
    model = Wrapper(model)
    embedding = model(current_file)
    labels = annotation.labels()
    for l, label in enumerate(labels):
        timeline = annotation.label_timeline(label, copy=False)

        # be more and more permissive until we have
        # at least one embedding for current speech turn
        for mode in ['strict', 'center', 'loose']:
            x = embedding.crop(timeline, mode=mode)
            if len(x) > 0:
                break

        # skip labels so small we don't have any embedding for it
        if len(x) < 1:
            continue

        #average speech turn embeddings
        x = np.mean(x, axis=0)

        #append reference to the references
        references.setdefault(label,[])
        references[label].append(x)
    return references

def load_pretrained_pipeline(train_dir: Path) -> Pipeline:
    """Load pretrained pipeline

    Parameters
    ----------
    train_dir : Path
        Path to training directory (i.e. the one that contains `params.yml`
        created by calling `pyannote-pipeline train ...`)

    Returns
    -------
    pipeline : Pipeline
        Pretrained pipeline
    """

    train_dir = Path(train_dir).expanduser().resolve(strict=True)

    config_yml = train_dir.parents[1] / "config.yml"
    with open(config_yml, "r") as fp:
        config = yaml.load(fp, Loader=yaml.SafeLoader)

    pipeline_name = config["pipeline"]["name"]
    Klass = get_class_by_name(
        pipeline_name, default_module_name="pyannote.audio.pipeline"
    )
    pipeline = Klass(**config["pipeline"].get("params", {}))

    return pipeline.load_params(train_dir / "params.yml")
