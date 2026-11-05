# Reliance shaping for vision — Waterbird

Coming soon.

This experiment applies reliance shaping to a frozen vision backbone (ResNet-50 and DINOv2) trained on the Waterbirds benchmark, where the background habitat perfectly predicts the bird label at training time.  Reliance shaping recovers ceiling worst-group accuracy on both backbones (0.462 → 0.940 for ResNet-50, 0.628 → 0.957 for DINOv2).  Three variants are provided: a linear head, a nonlinear head (where the seven reliance measures begin to differ), and a fully trainable backbone (where circumvention can occur).
