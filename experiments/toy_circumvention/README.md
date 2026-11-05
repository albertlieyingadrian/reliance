# Toy circumvention model

Coming soon.

This experiment implements the two-layer linear model from Section 6 of the paper, which explains geometrically how training against a probe changes the representation the probe measures.  The model has a fixed projection P that removes a concept direction from the activations, and we track what happens to the concept as training continues: whether it drifts to a new direction, gets erased from the layer, or survives as a product-code that linear probes can no longer read.

The setup requires no external data or pretrained weights and runs in seconds, making it a clean testbed for Propositions 1–4.
