# Tennis Gym

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


### Introduction

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.



### Solution

I solve this gym using multiple DDPG agents with a replay buffer. I did try to implement a priority replay buffer, but my experimentation were futile so I switched back to a regular replay buffer. For my actor and critic network I used to fully connected network. 

I used batch norm and weighted decay to regularize my model. I played with the number of layers, and number of neurons while monitoring the scores. 

I also applied gradient clipping to both the critic and the actor network to stabalize the training. Both agents (rackets?) were trained simultaneously. I added Ornstein–Uhlenbeck noise to enourage explorations and clipped the action values to be between -1 and 1. I used soft updates for updating the network models. I trained the model every 64 times to stablize the model convergence. I used 1e6 as my replay buffer size and batch the data with 128 samples. I used MSE to minimize the loss between predicted and target values. 


You can find the final pytorch models for the critic and actor model here "model/". 


Actor Model: 

    4 fully-connected linear layers where :
    
        1st layer is 256 neurons followed by ReLU and Batch Norm
        2nd layer 256 neurons followed by ReLU and Batch Norm
        3rd layer down-size it to 128 neurons followed by ReLU
        4th laayer down-size to number of action and followed by TanH


Critic Model:

    3 fully-connected linear layers where :
    
        1st layer is 256 neurons followed by ReLU and Batch Norm
        2nd layer 256 neurons followed by ReLU and Batch Norm
        3rd layer down-size it to 1 neron followed by TanH

I initialized the weights kaiming uniform initialization for first two layers and a uniform initalization for the last layer. I used Adam optimizer with L2 regularization. 

I also tried to train an ensemble of agent and taking a weighted action instead, but the computation was too expensive and I could not see an immediate improvement to the rewards. You can see that this agent easily beat the 0.5 rewards requierments averaging round 2. You can also find a video of the gym working at test time here, "snapshot/tennis_gym.mov". 

You can see that acheive and maintain an average score of bigger than 0.5 over 100 episode after episode 50. The score keeps going on to a maximum of 2.5 where the average score is around 2.0. 

![Alt text](/snapshot/training_score.png)

###  Future Work

As for future work, it would be nice to try out PPO agents instead of DDPG agents. I'm curious to see how well or worse an actor only network would work. I would also implement the priority queue that I had attempted to implement to see if it might help with stabalizing the scores.
