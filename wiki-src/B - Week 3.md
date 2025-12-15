## Week 3

**What went well this week?**
------------------------------
This week was productive, especially in terms of technical depth and problem-solving. I completed the remaining Week 2 notebooks, which significantly improved my understanding and directly supported my progress on Task 4 (dataset preparation). I finalized the patch-generation pipeline, validated the generated patches, and ensured mask correctness through sanity checks.

I also made meaningful progress on Task 5 by setting up the first model training experiments. Although the initial results were not strong, starting with a baseline U-Net helped me understand the limitations of the current setup and identify directions for improvement. Additionally, working in a quiet environment while traveling helped me stay focused and productive.

**What did you struggle with this week?**
------------------------------
The main challenges were related to debugging and infrastructure. I spent a considerable amount of time identifying why dataset variables were empty, which turned out to be caused by inconsistent folder structures. Model training was also challenging due to slow performance on my local machine. Attempts to move training to the server failed because of issues with data generators, which was frustrating and time-consuming.

**How did your efforts in applying iterative critical thinking this week
contribute to improving your approach to the project? (ILO 2.6)**
------------------------------
I consistently applied iterative critical thinking by questioning assumptions and revisiting earlier decisions. For example, when Task 4 initially failed, I examined the dataset structure instead of assuming the code logic was wrong. This helped me identify the real issue and adjust my approach.

I also critically evaluated peer feedback on annotations rather than accepting it blindly, using evidence and mentor validation to confirm my conclusions. During model training, I treated low performance not as failure but as feedback, leading me to consider alternative patch sizes, loss functions (e.g. Dice loss), and hardware solutions such as GPU acceleration. Reflecting on interruptions due to travel also helped me plan more proactively and maintain progress despite changes in routine.

**Based on this, what action points can you create for next week?**
------------------------------
1. Finalize and submit Task 1 annotations and ensure consistency across all images.
2. Set up GPU/CUDA support to enable more efficient model training.
3. Continue Task 5 by experimenting with different patch sizes and loss functions (e.g. Dice loss).
4. Improve robustness of data generators to support both local and server-based training.
5. Continue documenting reflections explicitly to demonstrate analytical thinking and learning progress.