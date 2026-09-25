"""Typed-decision models: a fine-tuned tower with a trained option scorer.

The file split is a guard rail, not a style. An automated experiment may rewrite
the EDITABLE modules and may not touch the PROTECTED ones, so a change to the
model, the data or the training recipe can never quietly become a change to how
it is scored. `rsijev/README.md` has the table.

    PROTECTED   contract.py  encode.py  evaluate.py  metrics.py  targets.py
    EDITABLE    arch.py      data.py    train.py     (fit.py: bookkeeping is not)
"""
__all__ = ["contract", "metrics", "arch", "data", "train"]
