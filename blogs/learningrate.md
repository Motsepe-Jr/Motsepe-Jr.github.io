---
layout: page
permalink: /blogs/gelu/index.html
title: "Evaluating Types of Learning Rates on Mr. Karpathy's GPT-2"
---

## Evaluating Learning Rate Scheduling Techniques on Mr. Karpathy's GPT-2

**Date:** August 02, 2024  
**Estimated Reading Time:** 20 min  
**Author:** Hector Motsepe

Recently, Mr. Andrej Karpathy trained GPT-2 from scratch using the Cosine Schedule learning rate. Inspired by this, I decided to test various learning rate scheduling techniques to see how they improve the performance, convergence speed, and training stability of AK's GPT-2 model.

### Understanding Learning Rates

- **Definition:** Learning rates determine the step size in an optimization algorithm at each iteration, guiding the movement towards the minimum of a loss function (Murphy, 2012).
- **Importance:** Learning rates influence how newly acquired information overrides old information, affecting how the model "learns."
- **Function:** The gradient of the loss determines the descent direction, while the learning rate determines the step size in that direction.

### Setting the Learning Rate

- **Trade-offs:** Balancing between the rate of convergence and overshooting is crucial.
  - **High Learning Rate:** Can cause the learning to jump over minima.
  - **Low Learning Rate:** Can lead to slow convergence or getting stuck in undesirable local minima.
- **Adaptation:** Finding a sweet spot or adapting the learning rate during training is essential.

### Techniques

- **Decay-Based Learning Rate:** Gradually reducing the learning rate over time allows the model to make larger updates initially and smaller updates as training progresses, avoiding overshooting around the optimum.
- **Step/Time-Based Learning Rate:** Altering the learning rate based on previous iterations.


### Experimentation


![Learning Rate Loss](/blogs/assets/learningrate/loss.png)

## Learning rates 



References 
- Murphy, Kevin P. (2012). Machine Learning: A Probabilistic Perspective. Cambridge: MIT Press. p. 247. ISBN 978-0-262-01802-9.