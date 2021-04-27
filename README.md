<!-- omit in toc -->
# Implementing Reinforcement Learning with Google Chrome's *Dinosaur Game*  

![Here's a dino trained with the Deep Q-Learning algorithm!](/assets/dino_game_dql.gif "Dino with Deep Q-Learning!")

In this repository, you'll find starter code and instructions for
implementing two different reinforcement learning algorithms
to train a bot to play Google Chrome's *Dinosaur Game*.
Specifically, the two algorithms used are
Q-Learning and Deep Q-Learning.


This repository is aimed at people who feel
relatively comfortable with Python but who may not be
super familiar with reinforcement learning.
Nevertheless, folks with prior experience in the field
may find this repository useful for developing other
algorithms and optimizations for the *Dinosaur Game*.

<!-- omit in toc -->
# Contents

- [Getting Started](#getting-started)
- [Installation Process](#installation-process)
  - [For Windows](#for-windows)
  - [For Mac & Linux](#for-mac--linux)
- [A Quick Intro to Reinforcement Learning](#a-quick-intro-to-reinforcement-learning)
- [Implementing Q-Learning](#implementing-q-learning)
  - [Step 1: Using an Epsilon-Greedy Strategy](#step-1-using-an-epsilon-greedy-strategy)
  - [Step 2: Updating the Q-Table with the Bellman Equation](#step-2-updating-the-q-table-with-the-bellman-equation)
  - [Step 3: Training & Testing Your Q-Learning Agent](#step-3-training--testing-your-q-learning-agent)
- [Implementing Deep Q-Learning](#implementing-deep-q-learning)
  - [Step 1: Setting up the Epsilon-Greedy Strategy](#step-1-setting-up-the-epsilon-greedy-strategy)
  - [Step 2: Using an Experience Replay Buffer](#step-2-using-an-experience-replay-buffer)
  - [Step 3: Updating the Main Neural Network](#step-3-updating-the-main-neural-network)
  - [Step 4: Updating the Target Neural Network](#step-4-updating-the-target-neural-network)
  - [Step 5: Training & Testing Your Deep Q-Learning Agent](#step-5-training--testing-your-deep-q-learning-agent)
- [Wrapping Up](#wrapping-up)
- [Acknowledgements](#acknowledgements)
- [References](#references)

# Getting Started

Below is an explanation of each of the files
that can be found in this repository:
- *dino_game.py*: This file contains the *DinoGame* class,
  the main game loop, and most of the game's logic. **This
  is the file that you will be running from your terminal in
  order to test/train your reinforcement learning agents.**
  To make your task a bit easier, this file can be run with
  the following optional arguments:
  > - *--train*: For signaling that you want to train a new agent.
  > - *--test*: For signaling that you want to test a saved agent.
  > - *--train-episodes*: For specifying the number of episodes
  that you would like to use to train your new agent(s).
  > - *--test-episodes*: For specifying the number of episodes
  that you would like to use to test your saved agent(s).
  > - *--train-display*: For signaling that you want to visualize
  each training episode.
  > - *--no-test-display*: For signaling that you *do not* want
  to visualize each test episode.
  > - *--ql*: For signaling that you want to train and/or test
  a Q-Learning agent.
  > - *--dql*: For signaling that you want to train and/or test
  a Deep Q-Learning agent.

- *ql.py*: This file contains the *QLearning* class which handles all of
  the logic for the Q-Learning agent. **In order to get your Q-Learning 
  implementation working, you'll need to follow the steps listed under the
  [Implementing Q-Learning](#implementing-q-learning) section
  to fill out the missing parts of this class's code.**

- *dql.py*: This file contains the *DQLearning* class which handles all of
  the logic for the Deep Q-Learning agent. **To get a working
  Deep Q-Learning implementation, you'll need to fill out the missing 
  parts of this class's code by following the steps listed under the
  [Implementing Deep Q-Learning](#implementing-deep-q-learning) section.**

- *variables.py*: This file contains the code that manually defines
  all of the parameters and variables used for the game logic and
  reinforcement learning agents. **If you ever would like to
  change different hyperparameters for your agents, you can easily
  do so by changing the corresponding values in this file.**

- *dino.py*: This file contains the code for the *Dino* class,
  which handles the dino's game logic.

- *obstacle.py*: This file contains the code for the *Obstacle* class,
  which handles generating, drawing, and moving the obstacles
  on the screen.

- *background.py*: This file contains the code for the *Ground*, 
  *Cloud*, and *Scoreboard* classes; these classes mostly 
  handle rendering the screen's background sprites.

# Installation Process

Before doing anything, you first need to make sure 
that you have [installed the most recent
version of python](https://www.python.org/downloads/). Once you've got that ready, 
please follow the installation steps below that correspond to your device's
operating system.

## For Windows

- [Fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) this repository.

- Open your terminal and change your working directory to be the folder where you want put the local
  version of your forked repository.

- [Clone](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
  your forked repository into your working directory.

- Change your working directory to be main folder of your cloned repository.

- Use the following terminal commands to 
  [create a virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/), 
  activate the virtual environment, and install the project's dependencies.
  ```
  py -m venv env
  .\env\Scripts\activate
  py -m pip install -r requirements.txt
  ```

- Test your installation by running the following command:
  ```
  py -m dino_game --ql --test
  ```
  After running this command, you should see a pop-up that shows the dinosaur running across
  the screen without responding to the moving obstacles.

- Once you've verified your installation, use the following command to deactivate the virtual
  environment:
  ```
  deactivate
  ```

**Please note** that whenever you would like to run the project's *dino_game.py* file,
you should always activate your virtual environment beforehand by using
the following terminal command:
```
.\env\Scripts\activate
```
Once you're done with running the files, be sure to run the following
terminal command to deactivate the virtual environment:
```
deactivate
```

## For Mac & Linux

- [Fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) this repository.

- Open your terminal and change your working directory to be the folder where you want put the local
  version of your forked repository.

- [Clone](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)
  your forked repository into your working directory.

- Change your working directory to be main folder of your cloned repository.

- Use the following terminal commands to 
  [create a virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/), 
  activate the virtual environment, and install the project's dependencies.
  ```
  python3 -m venv env
  source env/bin/activate
  python3 -m pip install -r requirements.txt
  ```

- Test your installation by running the following command:
  ```
  python3 -m dino_game --ql --test
  ```
  After running this command, you should see a pop-up that shows the dinosaur running across
  the screen without responding to the moving obstacles.

- Once you've verified your installation, use the following command to deactivate the virtual
  environment:
  ```
  deactivate
  ```

**Please note** that whenever you would like to run the project's *dino_game.py* file,
you should always activate your virtual environment using
the following terminal command:
```
source env/bin/activate
```
Once you're done with running the files, be sure to run the following
terminal command to deactivate the virtual environment:
```
deactivate
```

# A Quick Intro to Reinforcement Learning

Reinforcement learning is a specific branch of machine learning
that was originally inspired by the field of behavioral psychology.
Broadly, reinforcement learning is an iterative process that
generally exhibits the following cyclical structure:

- First, the reinforcement learning *agent* (or bot) 
  must observe the current *state* of its *environment*.

- Next, the agent uses some *policy* (or strategy) to
  determine the best expected *action* for it to take
  given the current state.

- Afterwards, the agent performs the best expected action
  and measures the true *reward* or *punishment* for
  taking that action.

- Then, the agent updates its policy based on the 
  performed action's true reward or punishment,
  before repeating the process all over again.

In terms of the code, rewards are simply positive values
while punishments are negative values. The larger the reward's
value, the better the action. The smaller (more negative)
the punishment's value, the worse the action.

If you'd like to learn more about reinforcement learning,
I strongly recommend reading 
[this article](https://smartlabai.medium.com/reinforcement-learning-algorithms-an-intuitive-overview-904e2dff5bbc)
by SmartLab AI.

# Implementing Q-Learning

When it comes to training a bot to play a video game, one of the most
common reinforcement learning algorithms used is Q-Learning.
Q-Learning works by maintaining and updating
values (known as Q-Values) for all possible state-action
pairs for the agent's environment. Each Q-Value represents
how good its corresponding action is expected to be given its
corresponding state. All Q-Values are stored in a table 
known as a Q-Table and are initialized to 0 at the beginning
of the training processes. As the agent gets more experience
interacting with its environment, the Q-Values are
iteratively updated using a function known as the 
[Bellman equation](https://www.mlq.ai/deep-reinforcement-learning-q-learning/#2-the-bellman-equation). 
This may be a lot to take in, but hopefully going through the process
of implementing your own Q-Learning agent will help you get a
better feel for how the algorithm works.

Before implementing Q-Learning for a game, a very important 
preliminary step involves determining which information about the game environment
will be useful for representing each state. Using too much information 
for each state will cause a Q-Learning agent to be very susceptible to noise
in the environment (as a result of
[overfitting](https://www.ibm.com/cloud/learn/overfitting#toc-what-is-ov-XiD4KBDG));
however, using too little information will cause the agent to 
make poor decisions (as a result of
[underfitting](https://www.ibm.com/cloud/learn/overfitting#toc-overfittin-7bQGsQfX)).
Since striking this balance can be quite tricky, the starter code already
handles setting up a sufficient state representation for you. 
This state representation corresponds to the nearest obstacle to 
the agent at a given timestep; in particular, each state is a tuple
containing 5 integers that represent the following information
about the nearest obstacle: its left-most horizontal position,
its bottom-most vertical position, its width, its height, and
its current travel speed.

Other important preliminary steps involve determining
how to represent each action and how to reward actions
during the training process. The starter code also handles
both of these steps for you. Specifically,
the code represents the agent's
three actions of running, jumping, and ducking with
integer values 0, 1, and 2 respectively. Additionally,
the starter code rewards any actions that cause the agent to
successfully avoid an obstacle and penalizes any
actions that cause the agent to crash into
obstacles.

Alright! With that background info out of the way,
you're ready to dive into the starter code in 
the "ql.py" file in the "rl_algorithms" subfolder
to implement your own Q-Learning agent.

## Step 1: Using an Epsilon-Greedy Strategy

As was previously mentioned, an agent's
Q-Table starts blank at the beginning of
the training process, which means that
it wouldn't really be useful for making
decisions at first. One approach to building
up the Q-Table uses what is known
as an Epsilon-Greedy Strategy.

At its core, an Epsilon-Greedy Strategy uses a
value between 0 and 1 (inclusive) known as *epsilon* to determine whether
the agent should perform a random
action or perform the best expected
action. Notably, *epsilon* is not a constant
value, but rather changes while the agent
is being trained. At the beginning of the training
process, the value of *epsilon* is higher so that
the inexperienced agent can *explore* the environment
by performing random actions. As the training process
continues and the agent gains more experience, the
value of *epsilon* gradually decreases in order to
encourage the agent to more frequently *exploit* its
prior experiences to perform the best expected actions.

To kick things off, look for the helper function named
``` get_random_action() ``` in the "ql.py" file.
In order to implement this
function all you need to do is the following:

> - Return a random value from the action space. Please
note that the action space is a list.

Once you get done with that function, you should navigate to the
```get_best_action()``` helper function. This function takes as input
a state tuple and should return the action corresponding to the highest
Q-Value in the Q-Table for that state. The steps to implement this function
are the following:

> - Get the index of the given state in the state space. Please keep in
mind that the state space is a list.
>
> - Use the state's index and the Q-Table to get the Q-Values for each action.
>
> - Return the action corresponding to the highest Q-Value. 
*Hint: you may find [NumPy's argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html?)
function helpful.*

With these two helper functions out of the way,
you're ready to tie it all together. To do this,
you should update the ```choose_action()``` so
that it does the following:

> - If the random value from [0, 1) is less than epsilon AND if the agent is training...
>   - Use the ```get_random_action()``` helper function to get a random action.
> 
> - Else...
>   - Use the ```get_best_action()``` helper function to return the best expected action for the given state.

## Step 2: Updating the Q-Table with the Bellman Equation

At this point, you should be ready to implement the code
that updates the agent's Q-Table. To do this, you
should find the take a look at the file's ``` update() ```
function. This function takes as input a state tuple,
an action integer, a Dino class object, and a next state tuple.
Keeping this in mind, you should fill out the rest of the function
so that it does the following:

> - Get the index of the current state in the state space. Please keep in
mind that the state space is a list.
> 
> - Using the state index, get the current Q-Value in the Q-Table 
for the current state and action.
>
> - Get the index of the next state in the state space.
> 
> - Using the next state index, get the highest Q-Value in the Q-Table
among all actions for the next state. 
*Hint: you may find 
[NumPy's amax](https://numpy.org/doc/stable/reference/generated/numpy.amax.html)
function helpful.*
>
> - Calculate the new Q-Value for the given state-action pair using the Bellman
equation:
>$$ \text{New Q-value} = \text{Old Q-Value} + \alpha \cdot ( \text{Reward} + \gamma \cdot \text{Max Q-Value for next state} - \text{Old Q-Value})$$
>
> - Replace the current Q-Value in the Q-Table for the current state-action 
pair with the new Q-Value.

## Step 3: Training & Testing Your Q-Learning Agent

After filling out all of the missing code for your Q-Learning agent's
helper functions, you should be able to start with the fun part:
training and testing the agent! To do this, you'll need to open
your terminal, navigate to this project's local directory on your
device, and activate the project's virtual environment.

If you are using a Windows device, you can then run the following
command:
```
py -m dino_game --ql --train --train-episodes=10000 --test --test-episodes=10
```

If you are using a Mac or Linux device, you should instead run the following
command:
```
python3 -m dino_game --dql --train --train-episodes=100 --test --test-episodes=10
```

This command trains your Q-Learning agent with 10000 episodes and tests your
agent for 10 episodes. Please keep in mind that these training episodes will
not be visualized (since the ```--train-display``` [optional argument](#getting-started)
wasn't used) and the test episodes will be visualized (since the ```--no-test-display```
[optional argument](#getting-started) wasn't used).

As is often the case when coding, you may encounter several bugs the
first time you run your Q-Learning implementation. If that happens,
resolving the compiler issues should be your best bet moving forward.
If you still encounter issues, you may want to review the
instructions in [step 1](#step-1-using-an-epsilon-greedy-strategy) and
[step 2](#step-2-updating-the-q-table-with-the-bellman-equation)
to see what might be going wrong.

If you're implementation runs with no issues, the Q-Learning dino should
behave something like this after being trained on 10000
episodes:
![Q-Learning agent after 10000 training episodes!](/assets/dino_game_ql10000.gif "Q-Learning agent after 10000 training episodes!")

**Don't forget** to deactivate your virtual environment when you are done training
and testing.

# Implementing Deep Q-Learning

Although the *Dinosaur Game* is relatively simple, its Q-Table
is actually quite large. For even more complex games (like 
[DOOM](https://www.freecodecamp.org/news/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8/)),
the size of the Q-Table would become unreasonably large and costly to store in memory.
This is where the Deep Q-Learning algorithm comes in.

Just like the name suggests, Deep Q-Learning is a modified version of Q-Learning.
The main difference between the two algorithms is that regular Q-Learning uses
a Q-Table whereas Deep Q-Learning uses a deep neural network to map
different states to actions.

For those who may be new to deep learning and/or deep neural networks,
I strongly recommend watching the [video series on deep learning
by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).
These videos should do an excellent job of getting you up to speed on the logic
behind Deep Q-Learning, which is crucial for completing
the next sections of this guide.

When training, Deep Q-Learning uses two nearly identical neural networks
to improve overall stability. The first is known as the
*main* neural network and is iteratively updated
using the agent's experiences. The second is known
as the *target* neural network and is updated
using the main neural network's weights.

The process for setting up neural networks can be quite
challenging to do from scratch. Fortunately several Python
packages exist to make this process straight-forward and relatively
simple. This project will use the 
[PyTorch](https://pytorch.org/docs/stable/index.html)
because it is easy to use and has built-in
compatibility with NumPy. Additionally, the
starter code already handles building both the main and
target neural networks for you.

Once you feel comfortable with 
deep learning and deep neural
networks, you should open up
the "dql.py" file to get started
with your first task of setting
up the Deep Q-Learning agent's Epsilon-Greedy
Strategy.

## Step 1: Setting up the Epsilon-Greedy Strategy

Assuming that you've completed the Q-Learning section
of this guide, setting up the Epsilon-Greedy Strategy
for the Deep Q-Learning agent should hopefully feel 
like a familiar task.

To start, first navigate to the ``` get_random_action() ```
helper function in the "dql.py" file.
Assuming you've already implemented the Q-Learning algorithm,
the code for this function is exactly the same as the code
you wrote for the same function in [step 1](#step-1-using-an-epsilon-greedy-strategy).

For now let's skip over the ```get_best_action()``` helper function
and jump to the ```choose_action()``` function instead. Just like before,
assuming you've already implemented Q-Learning, the code for this
function should be identical to your code for the same function from
[step 1](#step-1-using-an-epsilon-greedy-strategy)
of the Q-Learning section.

With all of that out of the way, let's now jump back up to the
``` get_best_action() ``` helper function. Unlike the other two functions
you just worked on, this function differs greatly from the regular Q-Learning
implementation because it should use the main neural network
to get the best action instead of using a Q-Table.

The ``` get_best_action() ``` function takes a state tuple
as input and should return the given state's best expected
action. Since the input information for a PyTorch neural
network should be a tensor, the first step for this
function is converting the given state tuple
into a tensor. Please note that the starter code already handles
this small step for you. Using this state tensor, fill in
the missing code to get the best expected action by taking
the following approach:

> - Put the main neural network in 
[evaluation mode](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval).
>
> - [With no gradient](https://pytorch.org/docs/stable/generated/torch.no_grad.html), 
use the main neural network and the state tensor to generate output values for each possible action.
Don't forget to [detach](https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach) 
the resulting tensor from the network.
>
> - Put the main neural network back into 
[training mode](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train).
>
> - Return the action corresponding to the element in the output tensor with the highest value. 
*Hint: you may find PyTorch's 
[argmax](https://pytorch.org/docs/stable/generated/torch.argmax.html#torch.argmax) and
[item](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.item) 
functions helpful.*

## Step 2: Using an Experience Replay Buffer

In addition to using an Epsilon-Greedy Strategy during training,
the Deep Q-Learning algorithm should also make use of something
known as an Experience Replay Buffer, which stores a large set
of recent experiences in memory. With this Replay Buffer, 
the code can train the main neural network on a 
random sample of experiences (known as a *minibatch*)
instead of training the network on chronological experiences.
The main motivation for using an Experience
Replay buffer relates to the ways in which
the sequential nature of chronological experiences
can introduce bias when training a deep neural network.
Taking a relatively small random sample of recent experiences
when training the main neural network
reduces this bias since the random sample won't simply consist
of experiences that came one after the other.

To set up some of the logic for the Experience Replay Buffer
in the code, you should now navigate to the
```update()``` function in this file. Before you code
anything though, let's first take some time to go over what the
starter code already handles for you.

First, the code declares a boolean variable named ```done```
that is ```True``` if the agent has crashed into an object
and ```False``` otherwise. This variable will be used later to
prevent the code from taking into account the next state
in cases where the agent has already crashed (since there
would be no next state in the game after crashing).

Next, the code uses the ```remember()``` helper function to add
the information corresponding to this most
recent experience to the Experience Replay Buffer.

Afterwards, the code checks if enough new
experiences have been added to the experience replay
buffer before updating any of the neural
networks. This helps improve the overall
stability of the network by preventing problems
with [overfitting](https://www.ibm.com/cloud/learn/overfitting)
the data.

Alright, with all of that out of the way, you
should now be able to implement the rest of this
function, so that it does the following:
> - If the length of the replay buffer is greater than
> the batch size...
> 
>   - Create a minibatch by taking a random sample from
> the replay buffer. Please make sure that the size of the
> minibatch is equal to the batch size variable.
> 
>   - Pass the minibatch to the ```update_main()``` helper function
> in order to update the main neural network.
> 
>   - Call the ```update_target()``` helper function to update
> the target neural network.

## Step 3: Updating the Main Neural Network

After you've set up the code that handles
using the Experience Replay Buffer, you should
be ready to implement the code responsible
for updating the main neural network.
To do this, you should navigate to the ```update_main()```
helper function.

This code takes as input a minibatch, which is
a list of tuples consisting of a state tuple, an action integer,
a reward value, a next state tuple, and done flag.
The starter code already handles looping through
each tuple in the minibatch. Additionally,
the code also handles converting the state and next
state tuples into their corresponding tensors.

With all of that in place, you should fill in
the missing code to do the following:

> - Put the main neural network in 
[training mode](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train).
>
> - Put the target neural network in 
[evaluation mode](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval).
>
> - Pass the state tensor through the main neural network
to get an output tensor. Please note that each value
in the tensor corresponds to one possible action.
>
> - [With no gradient](https://pytorch.org/docs/stable/generated/torch.no_grad.html), pass the next state tensor
into the target neural network to get the output tensor
for the next state. Once you get the next state's output
tensor, you should be sure to [detach it](https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach)
from the network.
>
> - Calculate the max value in the next state's output tensor.
*Hint: you may find PyTorch's 
[max](https://pytorch.org/docs/stable/generated/torch.max.html#torch.max) and
[item](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.item) functions helpful.*
> - Calculate the target reward for the current action by using the
following equation:
> $$\text{target reward} = \gamma \cdot \text{max value in the next state's output tensor} $$
> 
> - [Clone](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.clone)
the current state's output tensor.
> 
> - Replace the current action's value in the cloned tensor with the
calculated target reward.
> 
> - [Detach](https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach) the cloned tensor.
> 
> - Input the state's output tensor and the updated cloned 
tensor into a loss function to get the loss. 
For this part, you should first uncomment the following line:
>   ``` python
>   loss = torch.nn.MSELoss(predicted_tensor, target_tensor).to(DEVICE)
>   ```
>   Then you should replace the ```predicted_tensor``` and ```target_tensor```
variables with your state's output tensor and updated cloned tensor variables
respectively.
> 
> - Lastly, once you finish with all of the previous steps, uncomment
> the following lines in the function:
>   ``` python
>   optimizer.zero_grad()
>   loss.backward()
>   optimizer.step()
>   ```
>   In case you're curious, these lines of code use loss to update the main neural network's weights through 
[backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3).

## Step 4: Updating the Target Neural Network

Once you've updated the main neural network, you're
ready to work on the code that updates the target
target neural network. For this, you should make
your way to the ```update_target()``` helper function.

Although different approaches exist for doing this,
the starter code's approach updates the target neural
network's weights by setting it equal to a linear interpolation
between the main neural network's weights and the
target neural network's weights.

Since this can be tricky, the starter code already handles
looping through each matching pair of nodes in the two neural
networks. Your task is to implement the following:

> - Multiply the weight of the main neural network's current node with ```tau```.
> - Multiply the weight of the target neural network's current node with ```1 - tau```.
> - Set the value of the ```weight``` variable to be equal to the sum
> of the two resulting values from the previous steps.

## Step 5: Training & Testing Your Deep Q-Learning Agent

Now that you've filled out the missing code for your Deep Q-Learning agent's
helper functions, you're ready to start training and testing
your agent! To do this, you'll need to open your terminal,
navigate to this project's local directory on your
device, and activate the project's virtual environment.

If you are using a Windows device, you can then run the following
command:
```
py -m dino_game --dql --train --train-episodes=200 --test --test-episodes=10
```

If you are using a Mac or Linux device, you should instead run the following
command:
```
python3 -m dino_game --dql --train --train-episodes=200 --test --test-episodes=10
```

This command trains your Deep Q-Learning agent with 200 episodes and tests your
agent for 10 episodes. Please keep in mind that these training episodes will
not be visualized (since the ```--train-display``` [optional argument](#getting-started)
wasn't used) and the test episodes will be visualized (since the ```--no-test-display```
[optional argument](#getting-started) wasn't used).

As is often the case when coding, you may encounter several bugs the
first time you run your Deep Q-Learning implementation. If that happens,
you should first try to resolve all compiler errors and warnings. 
Please note that unlike regular Q-Learning, the Deep Q-Learning
algorithm doesn't not always converge (which means that it doesn't
always find a good strategy to respond to the environment).
If you are encountering issues like this, you may need to
tune the *alpha*, *gamma*, and/or *tau* hyperparameters that
are defined in the ```init_dql()``` function in the
"variables.py" file in the "rl_algorithms" folder.
If you continue encountering issues, you may want to review the
instructions in [step 1](#step-1-using-an-epsilon-greedy-strategy) and
[step 2](#step-2-updating-the-q-table-with-the-bellman-equation)
to double-check your implementation.

If your implementation runs with no issues, the Deep Q-Learning
dino should behave something like this after being trained on 200
episodes:
![Deep Q-Learning agent after 200 training episodes!](/assets/dino_game_dql200.gif "Deep Q-Learning agent after 200 training episodes!")

**Don't forget** to deactivate your virtual environment when you are done training
and testing.

# Wrapping Up

By this point, hopefully you have a working implementation and a solid
understanding of one (if not two) of the reinforcement learning algorithms
that are explained in this guide.

Both algorithms each have different pros and cons that make them
good for different games. For simple games like the *Dinosaur Game*
the Q-Learning algorithm would probably be the best approach because
Q-Learning has a much stronger guarantee of convergence (unlike
Deep Q-Learning). For more complex games, however, Deep Q-Learning
would probably be the best option because it can learn with fewer
iterations and scales very well as the size of the state spaces
increases drastically.

At the end of the day, the choice is yours! Other reinforcement
learning algorithms, like Policy Gradients and Double Deep Q-Learning,
are also great options to explore too.

Please feel free to use any of the code in this repository
for you own projects and extensions.

# Acknowledgements

I would like to thank Dr. David Walker and the "You Be the Prof"
Princeton Independent Work Seminar for their support as I developed
this project.

I would also like to thank [Shivam Shekhar](https://github.com/shivamshekhar/Chrome-T-Rex-Rush), 
[Shantanu Bhattacharyya](https://github.com/bhattacharyya/reach_circle), 
[Mauro Comi](https://github.com/maurock/snake-ga),
and [Unnat Singh](https://github.com/unnat5/deep-reinforcement-learning/tree/master/dqn).
This project draws on much of their work and would likely not have
been possible without their amazing tutorials and repositories.

# References

This project draws extensively from the following GitHub
repositories and resources:

> - For the *Dinosaur Game*: 
>   - https://github.com/shivamshekhar/Chrome-T-Rex-Rush
>   - https://chromedino.com/assets/offline-sprite-2x-black.png
> 
> - For the Q-Learning implementation: 
>   - https://github.com/bhattacharyya/reach_circle
>
> - For the Deep Q-Learning implementation: 
>   - https://github.com/maurock/snake-ga
>   - https://github.com/unnat5/deep-reinforcement-learning/tree/master/dqn
>   - https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/DQN_agents/DQN.py

The links to the official documentation of this project's dependencies 
are the following:

> - https://numpy.org/doc/stable/
>
> - https://pytorch.org/docs/stable/index.html
>
> - https://www.pygame.org/docs/

Some additional resources that were consulted when creating this project
are the following:

> - https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
>
> - https://smartlabai.medium.com/reinforcement-learning-algorithms-an-intuitive-overview-904e2dff5bbc
>
> - https://www.ibm.com/cloud/learn/overfitting
> 
> - https://www.mlq.ai/deep-reinforcement-learning-q-learning/
> 
> - https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
>
> - https://www.freecodecamp.org/news/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8/
>
> - https://blog.gofynd.com/building-a-deep-q-network-in-pytorch-fa1086aa5435
>
> - https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc

