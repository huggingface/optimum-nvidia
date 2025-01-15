from typing import List

import numpy as np


def assert_generated_partially_match(
    generated: List[int], gold: List[int], atol: float
):
    assert 0 < atol < 1

    max_overlap = min(generated.shape[0], gold.shape[0])
    generated, gold = np.asarray(generated), np.asarray(gold)
    matched = np.sum(generated[:max_overlap] == gold[:max_overlap])
    ratio = matched / max_overlap

    assert (1.0 - ratio) < atol, (
        f"generated contact overlapping items is below tolerance {atol} ({ratio})"
    )


def assert_generated_text_partially_match(generated: str, gold: str, atol: float):
    assert 0 < atol < 1

    generated, gold = np.asarray(generated.split()), np.asarray(gold.split())
    max_overlap = min(len(generated), len(gold))
    matched = np.sum(generated[:max_overlap] == gold[:max_overlap])
    ratio = matched / max_overlap

    assert (1.0 - ratio) < atol, (
        f"generated contact overlapping items is below tolerance {atol} ({ratio})"
    )
