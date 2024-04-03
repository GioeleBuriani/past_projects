# Control of OpenAI Gym Lunar Lander with Deep Q-Learning

<br>
<img src="./Lander%20-%20Animation.gif" width="300">
<br>

**Location**: Technische Universiteit Delft  
**Period**: Jun 2022 - Aug 2022  

## Context
Within the Bio-inspired Intelligence and learning for Aerospace Applications course, I had to make a free project regarding one of many branches of machine learning. I decided to choose reinforcement learning and apply it to the Lunar Lander environment of OpenAI Gym. Due to the continuity of the state space, I had to resort to deep reinforcement learning in order to have a neural network as a function approximator. In particular, following a brief literature review, I decided to implement Deep Q-Learning. The results were highly satisfactory, with a 100% success rate within 1000 training episodes.

## Project Description
This project centers on leveraging Deep Q-Learning for the navigation and landing of a lunar lander within a simulated environment, showcasing a sophisticated application of reinforcement learning to achieve precise and controlled landing on a designated zone. The project unfolds in the OpenAI Gym’s LunarLander-v2 environment, a dynamic and challenging platform that simulates the complexity of lunar surface landing with the objectives of safe descent, landing accuracy, and fuel efficiency.

The endeavor begins with an exploration of the environment's dynamics, such as gravitational pull, thruster power, and terrain features, setting a foundation for the learning algorithm. The core of the project utilizes a neural network to approximate the Q-function, enabling the agent to learn and adapt its actions based on the state of the environment and the feedback received in the form of rewards. Enhancements like experience replay and a target network are integrated to improve learning stability and efficiency.

Key challenges addressed include the tuning of hyperparameters such as the learning rate and discount factor, balancing exploration with exploitation to ensure comprehensive learning, and adapting the reward mechanism to encourage desirable landing behaviors. The project demonstrates significant learning progression, as evidenced by the increasing scores over episodes, culminating in the consistent achievement of successful and efficient landings.

In summary, this project exemplifies the application of deep reinforcement learning in solving complex control problems, illustrating the potential of neural networks in enabling autonomous systems to learn and perform tasks with high precision and reliability. Through iterative training and refinement, the lunar lander agent evolves into a model of efficiency, demonstrating the profound capabilities of artificial intelligence in aerospace applications.

## Files
- **Lander - Animation.gif**: Animation of a final experiment
- **Lander - Code.py**: Python code for the project
- **Lander - Final report.pdf**: Final report for the project
