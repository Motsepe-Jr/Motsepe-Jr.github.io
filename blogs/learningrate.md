---
layout: page
permalink: /blogs/gelu/index.html
title: "Evaluating Types of Learning Rates on Mr. Karpathy's GPT-2"
---

## Evaluating Learning Rate Scheduling Techniques on Mr. Karpathy's GPT-2

**Date:** August 02, 2024  | **Estimated Reading Time:** 20 min   | **Author:** Hector Motsepe

Recently, Mr. Andrej Karpathy trained GPT-2 from scratch using the Cosine Schedule learning rate. Inspired by this, I decided to test various learning rate scheduling techniques to see how they improve the performance, convergence speed, and training stability of AK's GPT-2 model. Before we get into the experimentation, lets build the intuition first.

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
<p style="text-align: center;">Fig 1: Training Loss over Training Step For different types of Learning rate Scheduler</p>

The above graph indicates that the training loss over training step for different learning rate schedulers. We see that **ConstantLR** lead to slow convergence, and doesn't reach the lowest loss as  it doesnt adapt to the changing curvature of the loss landscape. As shown in the below code the learning rate stay constant across the entire training loop. 

```python
@dataclass
class ConstantLR:
    max_lr: float = 3e-4

    def __call__(self, iteration: int) -> float:
        return self.max_lr
```

The **StepLR** start with a high learning rate and then reduce it at each itercation, therfore allowing for finer adjustements and faster intial progress. The **StepLR** reduces the learning rate by a decay factor "gamma". Think of it as gamma determinming our decay rate at each iteration. Based on the above grpah the learning rate reaches Final Loss of 5.151307. As shown below gamma is a hyperameter, we need to manually determine how we should  decay our learning rate this can affect our convergence speed.

```python
@dataclass
class StepLR:
    max_lr: float = 3e-4
    max_iters: int = 1000
    gamma: float = 0.95

    def __call__(self, iteration: int) -> float:
        return self.max_lr * (self.gamma ** (iteration // self.max_iters))
```
The **MultiStepLR** starts with higher learning rate and then reduce it after reaching specific step "milestones". Unlike StepLR the learning can stay unchanged until reaching the next milstones. Gamma is "decay factor" is a hyperparameter requires manual tuning. 

```python
@dataclass
class MultiStepLR:
    max_lr: float = 3e-4
    max_iters: int = 50
    gamma: float = 0.95
    milestones: List[int] = (15, 30, 45)

    def __call__(self, iteration: int) -> float:
        return self.max_lr * (self.gamma ** sum([iteration >= m for m in self.milestones]))
```

![Learning Rate Loss](/blogs/assets/learningrate/schedules.png)
<p style="text-align: center;">Fig 2: Learning Rate Schedules</p>

References 
- Murphy, Kevin P. (2012). Machine Learning: A Probabilistic Perspective. Cambridge: MIT Press. p. 247. ISBN 978-0-262-01802-9.