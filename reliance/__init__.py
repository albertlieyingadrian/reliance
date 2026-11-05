"""Shaping what a model expresses and what it relies on.

Two specifications, one equation each.

Expression shaping (Eq. 2 of the paper) asks: does the model produce activations or
outputs that contain an unwanted concept?  It costs rollouts that express the concept
and samples them less often.  The cost is computed in expression.py.

Reliance shaping (Eq. 4) asks: does the model's decision change when the concept
changes?  It penalizes the attribution of the readout onto a direction v and pulls
the decision away from the shortcut.  The measures are in dependence.py.

Finding the direction v that the concept lives on is the shared first step.

    contrast_subspaces.py  find v from labelled activations: dom, paired_svd, pls, ...
    geometric_probes.py    find v by fitting a reader: linear_probe, svm, ...
    dependence.py          reliance measures as penalties: gradinput, banzhaf, ...
    expression.py          expression cost from per-step concept scores
    intervene.py           rewrite the activation directly: project_out, shuffle, ...
    lagrangian.py          DualRate -- sets lambda from gradient norms without a sweep
    layout.py              activations in and out of one canonical (n, p, c) form
"""
