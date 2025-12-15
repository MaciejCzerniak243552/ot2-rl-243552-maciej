## Week 4

**What went well this week?**
------------------------------
This week was highly productive and marked a major technical breakthrough. After resolving CUDA and cuDNN compatibility issues, I significantly accelerated model training and was able to run full multiclass experiments efficiently. Completing the end-to-end pipeline was an important milestone, as it allowed me to iterate across tasks rather than working in isolation.

I achieved strong results in Task 5 after performing proper class-wise evaluation, identifying realistic performance for the root class (80–85% F1). Progress in Task 6 was also substantial: through multiple iterations, I developed a morphology-based segmentation pipeline that produced reliable individual root segmentation. In Task 7, refining the RSA extraction logic led to a dramatic improvement in performance, moving from an initial weak baseline to a top-ranking Kaggle score. Overall, I successfully translated iterative experimentation into measurable performance gains.

**What did you struggle with this week?**
------------------------------
The main struggles were related to environment setup and evaluation complexity. Configuring CUDA correctly was time-consuming and highlighted how fragile deep learning environments can be. Model evaluation was also more complex than expected, particularly due to memory constraints and the need to compute metrics incrementally.

Task 6 posed conceptual challenges, as some approaches from the self-study materials were incompatible with my model outputs. Selecting individual roots reliably proved difficult, especially when plant sizes varied. Additionally, the intensity of the week led to fatigue, and I eventually needed to take a break to avoid burnout.

**How did your efforts in applying iterative critical thinking this week
contribute to improving your approach to the project? (ILO 2.6)**
------------------------------
When the model initially produced an unrealistically high overall F1 score, I investigated further by performing class-wise evaluation, which revealed more meaningful insights. This reinforced the importance of interpreting metrics critically rather than relying on aggregate scores.

In Task 6, repeated failures forced me to abandon the assumption that size-based blob selection was sufficient. By iteratively refining the pipeline—adjusting morphology operations, changing selection logic, and involving the full mask—I arrived at a more robust solution. In Task 7, I used baseline submissions strategically to test assumptions early, then refined the algorithm based on observed weaknesses, leading to a significant performance jump. These iterations helped me build solutions incrementally while learning from each failure.

**Based on this, what action points can you create for next week?**
------------------------------
1. Reduce prediction noise from Task 5 to improve downstream segmentation in Task 6.
2. Further refine individual root selection criteria to handle plants of varying sizes more robustly.
3. Continue improving RSA extraction with biologically meaningful definitions of root length.
4. Balance intensive experimentation with planned breaks to maintain productivity and avoid burnout.