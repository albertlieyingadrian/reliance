# Reliance shaping for sycophancy

Coming soon.

This experiment applies reliance shaping to control sycophancy during the DPO phase of post-training, as described in Section 3 of the paper.  A language model trained with RLHF learns to agree with the user's stated preference rather than give the correct answer, because the preference data reflects this pattern on average.  Reliance shaping locates the sycophancy direction in the residual stream and penalizes reliance on it during DPO, reducing sycophancy across models from 7B to 32B parameters while preserving general capabilities.  Data filtering alone does not reach the same target.
