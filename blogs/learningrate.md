---
layout: page
permalink: /blogs/gelu/index.html
title: "Evaluating Types of Learning Rates on Mr. Karpathy's GPT-2"
---

# Evaluating Learning Rates Scheduling Techniques on Mr. Karpathy's GPT-2

**Date:** August 02, 2024 | **Estimated Reading Time:** 20 min | **Author:** Hector Motsepe

Recently, Mr. Andrej karpathy trained GPT-2 from scratch using the Consine Schedule learning rate, so I thought it would be cool to test different learning rate scheduling technqiues and compare how they improve the AK's GPT-2 model performance, convergence speed and training stability. Before we get ahead of ourself, lets build the intuition first. 

Learning rate determine the step size in an optimization algorithm at each iteration while moving toward a minimum of a loss function (Murphy, 2012). It represent the speed at which deep learning model "learns" as it influnces to what extend newly aquired information overides old information. The gradient of the loss determine the descent direction, and the learning rate determine how big a step is taken in that direction. 

Setting the learning rate comes with trade off between the rate of convergence and overshooting. High learning rate will make the learning jump over minima but low learning rate will take time to converge or even get stuck in an undesiracblke local minima. So, we need to find a sweet spot or adapt the learning rate during the training steps. Research scientist have developed come up decay based learning rate where we gradually reduce learning rate over time, and allow the model to make larger updates initially, and then refine its parameter with smaller updates as training progress. This avoid over shooting around the optimum. One the other hand Step/Time based Alter learning rate based on the previous learning rate of the previous time iteration. 

![Learning Rate Loss](/blogs/assets/learningrate/loss.png)



References 
- Murphy, Kevin P. (2012). Machine Learning: A Probabilistic Perspective. Cambridge: MIT Press. p. 247. ISBN 978-0-262-01802-9.