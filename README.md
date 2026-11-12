<div align="center">

# Shaping & Monitoring Models through their Internal Representations

**Thomas Fel\* · Matthew Kowal\* · Matt Feiszli\* · Satchel Grant\* · Andrew Lee\* · Chris Earls\***

Fatih Dinc · Vasudev Shyam · Usha Bhalla · Leon Bergen · Siddharth Boppana · Ching Fang
Michael Byun · Tristan Gallent · Nicholas Wang · Tal Haklay · Dron Hazra
Neehar Kondapaneni · Connor Watts · Ryan Panwar · Thomas Dooms · Jack Merullo · Atticus Geiger · Ekdeep Singh Lubana

**Thomas McGrath† · Owen Lewis†**

<sub>\*Core contributors &nbsp;·&nbsp; †Equal senior contribution</sub>

<img src="assets/goodfire_logo.png" height="40" alt="Goodfire"/>

</div>

---

<img src="assets/shaping_waterbird.gif" width="100%" alt="Shaping and monitoring a model through its internal representations"/>

---

We propose Mechanistic Shaping, a unified framework for shaping models through their internal representations, applied during training so that models follow specifications about which concepts they *express* — measured via their outputs or internal representations — and also which concepts they *rely on* — what they use to make their decisions.  Across vision, medical imaging, and language, Mechanistic Shaping substantially reduces targeted behaviors while preserving task performance.  We also study whether shaped concepts remain monitorable, characterize three geometric mechanisms of circumvention (drift, erasure, nonlinear encoding), and introduce a hierarchy of monitoring levels that tells these forms apart.

---

## Reliance and Expression

<img src="assets/reliance_vs_expression.png" width="100%" alt="Shaping reliance vs shaping expression"/>

A **reliance** specification asks whether the model's answer would change if the concept changed — it constrains what the readout depends on, not what the layer contains.  An **expression** specification constrains what the model's activations or outputs contain — a rollout that expresses a harmful concept incurs a cost even if the model ultimately refuses.  The two specifications are independent: a model can encode harmfulness while refusing to produce it, or rely on gender while never mentioning it.

Both are stated as a penalty added to the task objective:

```
Reliance:    ω(θ) = E[ ν(z, v)² ]
Expression:  ω(θ) = E[ Σ_t  γ^(t−1) ρ(g(z_t)) ]
```

where `v` is a direction in activation space for the unwanted concept, `ν(z, v)` is an attribution measure (how much the readout changes when `v` changes), `g(z_t)` is a concept score at step `t`, and `ρ` is a nonneg function.  Finding `v` is the shared first step: `contrast_subspaces.py` estimates it from paired activations (the same input with and without the concept).  The choice of penalty measure matters far less than the direction — across the seven measures in `dependence.py`, the spread in worst-group accuracy is about 0.003 once the direction is correct.

---

## Code

```
reliance/
    contrast_subspaces.py   find v from paired activations: dom, paired_svd, pls, ...
    geometric_probes.py     find v by fitting a reader: linear_probe, svm, ...
    dependence.py           reliance penalties: gradinput, banzhaf, smoothgrad, ...
    expression.py           expression cost aggregated over trajectory steps
    intervene.py            rewrite the activation directly: project_out, shuffle
    lagrangian.py           DualRate — sets λ from gradient norms; name a target, not a value
    layout.py               activations in and out of one canonical (n, p, c) form
```

**Reliance shaping** — penalize how much the readout depends on `v`:

```python
from reliance import dependence as D
from reliance.contrast_subspaces import dom

v    = dom(activations_with_concept, activations_without)[0]   # (1, c) line -> direction (c,)
z    = tap[layer]                        # hooked activation (n, p, c)
pen  = D.gradinput(readout, z, v).pow(2).mean()
loss = cross_entropy(model(x), y) + lam * pen
loss.backward()
```

**Expression shaping** — cost rollouts that express the concept:

```python
from reliance import expression as E

scores        = E.score(trajectory_activations, v)          # (T,)
cost          = E.expression_cost(scores, gamma=0.9, reduce='max')
reward_shaped = reward - lam * cost
```

---

## Install

```bash
git clone https://github.com/goodfire-ai/reliance.git
cd reliance && pip install -e ".[plot,dev]"
pytest -m "not slow"
```

---

## Experiments

| Domain | Setting | Result |
|---|---|---|
| Vision — Waterbird | Frozen DINOv2, perfectly confounded | 0.628 → **0.957** worst-group (ceiling 0.966) |
| Vision — Waterbird | Frozen ResNet-50 | 0.462 → **0.940** worst-group (ceiling 0.937) |
| Language — Sycophancy | Reliance shaping during DPO, 7B–32B | Reaches sycophancy target data filtering cannot |
| RL — Reward hacking | Expression shaping in GRPO (Countdown) | Suppresses peeking; transfers to Chess |
| RL — Harmlessness | Expression shaping on frozen base (DoM) | JBB 0.70 → 0.01 at matched utility |

Code for the vision experiments is in `experiments/waterbird/`.  Language and RL experiments are coming soon.

---

## Citation

```bibtex
@article{fel2025shaping,
  title   = {Shaping \& Monitoring Models through their Internal Representations},
  author  = {Fel, Thomas and Kowal, Matthew and Feiszli, Matt and Grant, Satchel and Lee, Andrew and Earls, Chris and
             Dinc, Fatih and Shyam, Vasudev and Bhalla, Usha and Bergen, Leon and Boppana, Siddharth and Fang, Ching and
             Byun, Michael and Gallent, Tristan and Wang, Nicholas and Haklay, Tal and Hazra, Dron and
             Kondapaneni, Neehar and Watts, Connor and Panwar, Ryan and Dooms, Thomas and
             Merullo, Jack and Geiger, Atticus and Lubana, Ekdeep Singh and
             McGrath, Thomas and Lewis, Owen},
  journal = {arXiv preprint},
  year    = {2025},
}
```
