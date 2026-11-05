# Reliance shaping for medical imaging

Coming soon.

This experiment applies reliance shaping to a pneumothorax classifier trained on CheXpert (Irvin et al., 2019), where the model learns to rely on the presence of a chest drain rather than the pathology itself — a shortcut that is spuriously correlated with the label in the training data.  Section 2.3 of the paper also shows how the concept direction can be identified autonomously, without a human naming the shortcut: an LLM agent screens candidate SAE features and selects the ones that correspond to the drain.

The CheXpert dataset requires a data use agreement and is not distributed here.  Instructions for access are at [stanfordaimi.github.io/CheXpert-v1-dataset](https://stanfordaimi.github.io/CheXpert-v1-dataset).
