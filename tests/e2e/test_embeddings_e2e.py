"""
assumes models are available at data/models
change seeds at the appropriate places to avoid backend caching
set --longrun flag when running pytest to run these tests
"""
import json
import os
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

import torch

import fsspec
from fsspec.implementations.dirfs import DirFileSystem

import pytest

from esm import ESM2, ProteinBertModel
from esm.pretrained import load_model_and_alphabet_core, _has_regression_weights

from protembed.alphabets import Uniprot21
from protembed.datasets import pad_tensor_1d
from protembed.factory import ProtembedModelLoader

import openprotein
from openprotein import OpenProtein
from openprotein.api.embedding import SVDModel
from tests.utils.svd import TorchLowRankSVDTransform


ALPHABET = Uniprot21()


def load_model_and_alphabet_local(model_location, device):
    """Load from local path. The regression weights need to be co-located"""
    model_location = Path(model_location)
    model_data = torch.load(str(model_location), map_location=device)
    model_name = model_location.stem
    if _has_regression_weights(model_name):
        regression_location = str(model_location.with_suffix("")) + "-contact-regression.pt"
        regression_data = torch.load(regression_location, map_location=device)
    else:
        regression_data = None
    return load_model_and_alphabet_core(model_name, model_data, regression_data)


@pytest.fixture()
def session() -> OpenProtein:
    with open("secrets.config", "r") as f:
        config = json.load(f)
    return openprotein.connect(
        config["username"],
        config["password"],
        backend="https://api.openprotein.ai/api/",
    )


@pytest.fixture()
def loader() -> ProtembedModelLoader:
    root_fs = fsspec.filesystem('file')
    dir_fs = DirFileSystem("data/models", root_fs)
    return ProtembedModelLoader(dir_fs)


@pytest.fixture()
def sequences() -> list[bytes]:
    rng = np.random.default_rng(188501)
    return [
        ALPHABET.decode(rng.integers(
            low=0,
            high=21,
            size=rng.integers(250, 500),
        ))
        for _ in range(5)
    ]


@pytest.fixture()
def same_length_sequences() -> list[bytes]:
    rng = np.random.default_rng(376735)
    return [
        ALPHABET.decode(rng.integers(low=0, high=21, size=331))
        for _ in range(5)
    ]


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
    sequences: list[bytes],
):
    print("testing...", model_id)
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
    if isinstance(attn, list):
        # TODO: kinda hacky. doing this b/c inferface of prosst and rotaformer are not
        # the same
        attn = torch.stack(attn, dim=1)
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
        random_mean_delta = np.abs(
            result[s] - actual[np.random.permutation(len(actual))]
        ).mean()
        print(
            "attn",
            mean_delta,
            random_mean_delta,
            random_mean_delta / mean_delta,
        )
        assert np.abs(result[s] - actual).mean() < 1e-4

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


@pytest.mark.longrun
@pytest.mark.parametrize(
    "model_id", ["esm1b_t33_650M_UR50S", "esm1v_t33_650M_UR90S_1", "esm2_t6_8M_UR50D"]
)
@torch.inference_mode()
def test_esm(session: OpenProtein, model_id: str, sequences: list[bytes]):
    print("testing...", model_id)
    device = (
        torch.device("cpu")  # using cpu in case of low vram
        if model_id != "esm2_t6_8M_UR50D" else torch.device("cuda")
    )
    local_model: Union[ESM2, ProteinBertModel]
    model_dir = "data/models"
    model_pt_path = os.path.join(model_dir, f"{model_id}.pt")
    local_model, alphabet = load_model_and_alphabet_local(
        model_pt_path, device
    )
    batch_converter = alphabet.get_batch_converter()
    local_model = local_model.eval()  # disables dropout for deterministic results
    if isinstance(local_model, ESM2):
        # half precision inference should be safe, per https://github.com/facebookresearch/esm/issues/283#issuecomment-1254283417
        local_model = local_model.half()
    local_model = local_model.to(device)
    can_predict_contacts = _has_regression_weights(model_id)

    _, _, batch_tokens = batch_converter(list(zip(
        [f"{i}" for i in range(len(sequences))],
        [s.decode().replace("X", "<mask>") for s in sequences],
    )))
    results = local_model(
        batch_tokens.to(device),
        repr_layers=[local_model.num_layers],
        need_head_weights=True,
        return_contacts=can_predict_contacts,
    )

    embeddings = results["representations"][local_model.num_layers].float()
    attn = results["attentions"].float()
    logits = results["logits"].float()
    if can_predict_contacts:
        contacts = results["contacts"].float()
    else:
        contacts = None

    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    embeddings = [
        embeddings[i, :tokens_len]
        for i, tokens_len in enumerate(batch_lens)
    ]
    mean_embeddings = torch.vstack([e[1:-1].mean(dim=0) for e in embeddings])
    sum_embeddings = torch.vstack([e[1:-1].sum(dim=0) for e in embeddings])
    attn = [
        attn[i, -1, :, :tokens_len, :tokens_len]
        for i, tokens_len in enumerate(batch_lens)
    ]
    logits = [
        logits[i, :tokens_len]
        for i, tokens_len in enumerate(batch_lens)
    ]
    if contacts is not None:
        contacts = [
            contacts[i, :tokens_len-2, :tokens_len-2]
            for i, tokens_len in enumerate(batch_lens)
        ]
    else:
        contacts = None

    embeddings = [x.float().cpu().numpy() for x in embeddings]
    mean_embeddings = [x.float().cpu().numpy() for x in mean_embeddings]
    sum_embeddings = [x.float().cpu().numpy() for x in sum_embeddings]
    attn = [x.float().cpu().numpy() for x in attn]
    logits = [x.float().cpu().numpy() for x in logits]
    contacts = (
        [x.float().cpu().numpy() for x in contacts]
        if contacts is not None else None
    )

    model = session.embedding.get_model(model_id=model_id)
    future = model.attn(sequences)
    time.sleep(1)
    future.wait_until_done()
    result = {s: x for s, x in future.get()}
    for s, actual in zip(sequences, attn):
        mean_delta = np.abs(result[s] - actual).mean()
        random_mean_delta = np.abs(
            result[s] - actual[np.random.permutation(len(actual))]
        ).mean()
        print(
            "attn",
            mean_delta,
            random_mean_delta,
            random_mean_delta / mean_delta,
        )
        assert np.abs(result[s] - actual).mean() < 1e-4

    for reduction in [None, "MEAN", "SUM"]:
        future = model.embed(sequences, reduction=reduction)
        time.sleep(1)
        future.wait_until_done()
        result = {s: x for s, x in future.get()}
        for i, s in enumerate(sequences):
            if reduction is None:
                actual = embeddings[i]
            elif reduction == "MEAN":
                actual = mean_embeddings[i]
            elif reduction == "SUM":
                # compare means to average out errors
                actual = sum_embeddings[i] / len(s)
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


@pytest.mark.longrun
@pytest.mark.parametrize("reduction", [None, "MEAN", "SUM"])
@pytest.mark.parametrize("random_state,should_fail", [(47, False), (100, True)])
def test_svd(
    session: OpenProtein,
    same_length_sequences: list[bytes],
    reduction: Optional[str],
    random_state: int,
    should_fail: bool,
):
    print("testing svd...", reduction, random_state, should_fail)
    sequences = same_length_sequences
    # this is an extremely strong test!
    # it depends on the svd random_state being the same
    model_id = "prot-seq"
    n_components = 1024
    model = session.embedding.get_model(model_id=model_id)

    # get embeddings to svd
    future = model.embed(sequences, reduction=reduction)
    time.sleep(1)
    future.wait_until_done()
    result = {s: x for s, x in future.get()}
    embeddings = np.stack([result[s] for s in sequences])
    if embeddings.ndim > 2:
        assert embeddings.ndim == 3
        embeddings = embeddings.reshape(len(sequences), -1)
    assert embeddings.ndim == 2

    # compute svd locally
    local_svd = TorchLowRankSVDTransform(
        n_components=n_components, random_state=random_state, device="cpu"
    )
    reduced_embeddings = local_svd.fit_transform(
        torch.from_numpy(embeddings).float()
    ).cpu().numpy()

    # get svd from remote
    svd: SVDModel = model.fit_svd(sequences, n_components=n_components, reduction=reduction)
    time.sleep(1)
    svd.get_job().wait_until_done(session=session)
    future = svd.embed(sequences)
    time.sleep(1)
    future.wait_until_done()
    result = {s: x for s, x in future.get()}
    for s, actual in zip(sequences, reduced_embeddings):
        mean_delta = np.abs(result[s] - actual).mean()
        random_mean_delta = np.abs(
            result[s] - actual[np.random.permutation(len(actual))]
        ).mean()
        print(
            "svd embed",
            mean_delta,
            random_mean_delta,
            random_mean_delta / mean_delta,
        )
        if not should_fail:
            assert random_mean_delta / mean_delta > 1e4
        else:
            assert random_mean_delta / mean_delta < 1e2
