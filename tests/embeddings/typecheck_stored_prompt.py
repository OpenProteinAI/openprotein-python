"""A stored prompt is a prompt to the PoET models, statically."""

from openprotein.data.prompts import Prompt
from openprotein.embeddings.poet import PoETModel
from openprotein.embeddings.poet2 import PoET2Model


def stored_prompt_inputs(poet: PoETModel, poet2: PoET2Model, prompt: Prompt) -> None:
    poet.embed([b"MKT"], prompt=prompt)
    poet.score([b"MKT"], prompt=prompt)
    poet.generate(prompt=prompt)
    poet2.embed([b"MKT"], prompt=prompt)
    poet2.score([b"MKT"], prompt=prompt)
    poet2.generate(prompt=prompt)
