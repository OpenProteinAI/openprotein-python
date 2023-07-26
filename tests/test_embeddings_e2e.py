"""
assumes models are available at data/models
change seeds at the appropriate places to avoid backend caching
set --longrun flag when running pytest to run these tests
"""
import json
import time

import numpy as np

import torch

import fsspec
from fsspec.implementations.dirfs import DirFileSystem

import pytest

from protembed.alphabets import Uniprot21
from protembed.datasets import pad_tensor_1d
from protembed.factory import ProtembedModelLoader

import openprotein
from openprotein import OpenProtein


ALPHABET = Uniprot21()


@pytest.fixture()
def session() -> OpenProtein:
    with open("secrets.config", "r") as f:
        config = json.load(f)
    return openprotein.connect(
        config["username"],
        config["password"],
        backend="https://dev.api.openprotein.ai/api/",
    )


@pytest.fixture()
def loader() -> ProtembedModelLoader:
    root_fs = fsspec.filesystem('file')
    dir_fs = DirFileSystem("data/models", root_fs)
    return ProtembedModelLoader(dir_fs)


@pytest.mark.longrun
@pytest.mark.parametrize("local_model_id,model_id", [
    ("prosst", "prot-seq"),
    ("rotaprot-seq-900m-uniref90-v1", "rotaprot-large-uniref90-ft"),
])
@torch.inference_mode()
def test_protembed(
    loader: ProtembedModelLoader,
    local_model_id: str,
    session: OpenProtein,
    model_id: str,
):
    print("testing...", model_id)
    np.random.seed(188256)
    sequences = [
        ALPHABET.decode(np.random.randint(
            low=0,
            high=21,
            size=np.random.randint(250, 500),
        ))
        for _ in range(5)
    ]

    local_model = loader.load_model(local_model_id, device=torch.device("cuda"))
    model = session.embedding.get_model(model_id=model_id)

    sequences_as_idxs, mask = pad_tensor_1d(
        [torch.from_numpy(ALPHABET.encode(s)).cuda().long() for s in sequences],
        ALPHABET.mask_token,
        return_padding=True,
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        _, attn = local_model.embed(
            sequences_as_idxs, padding_mask=mask, return_attention=True
        )
        embeddings = local_model.embed(sequences_as_idxs, padding_mask=mask)
        logits = local_model.logits(embeddings)
    attn = [
        x.float().cpu().numpy()[-1][:, :len(s), :len(s)]
        for s, x in zip(sequences, attn)
    ]
    embeddings = [
        x.float().cpu().numpy()[:len(s)]
        for s, x in zip(sequences, embeddings)
    ]
    logits = [x.float().cpu().numpy()[:len(s)] for s, x in zip(sequences, logits)]

    # we can't really make these difference tests too stringent, probably due to
    # numerical precision issues (fp16 may be particuarly problematic)
    future = model.attn(sequences)
    time.sleep(1)
    future.wait_until_done()
    result = {s: x for s, x in future.get()}
    for s, actual in zip(sequences, attn):
        mean_delta = np.abs(result[s] - actual).mean()
        print("attn", mean_delta)
        assert np.abs(result[s] - actual).mean() < 1e-2

    for reduction in [None, "MEAN", "SUM"]:
        future = model.embed(sequences, reduction=reduction)
        time.sleep(1)
        future.wait_until_done()
        result = {s: x for s, x in future.get()}
        for s, actual in zip(sequences, embeddings):
            if reduction == "MEAN":
                actual = actual.mean(axis=0)
            elif reduction == "SUM":
                # compare means to average out errors
                actual = actual.mean(axis=0)
                result[s] = result[s] / len(s)
            mean_delta = np.abs(result[s] - actual).mean()
            print("embed", reduction, mean_delta)
            assert np.abs(result[s] - actual).mean() < 1e-2

    future = model.logits(sequences)
    time.sleep(1)
    future.wait_until_done()
    result = {s: x for s, x in future.get()}
    for s, actual in zip(sequences, logits):
        mean_delta = np.abs(result[s] - actual).mean()
        print("logits", mean_delta)
        assert np.abs(result[s] - actual).mean() < 1e-2
