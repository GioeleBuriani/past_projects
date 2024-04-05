# Application of a neural network to drive a race car

<br>
<img src="./NNcar%20-%20Animation.gif" width="300">
<br>

**Location**: Technische Universiteit Delft  
**Period**: Sep 2021 - Nov 2021  
**Collaborators**: Edoardo Panichi

## Context
Within the Machine Learning for Robotics course, we designed a neural network classifier that identified actions for a car based on the surroundings. We were given a series of actions (accelerate, turn left, turn right, brake, do nothing) associated with specific moments within a car race in terms of surroundings. By training a classifier on that data, we had to create a system that could understand which action to perform at each moment in different race tracks (with different surroundings) in order to drive the car correctly and as fast as possible.

## Project Description
This project embarks on the exploration of planetary surface classification using a hybrid approach combining Convolutional Neural Networks (CNNs) and Random Forest classifiers, set within a simulated environment across multiple planets. The crux of the project lies in the development and refinement of a machine learning model capable of distinguishing planetary surfaces, utilizing images from Earth, Mars, and Saturn to train and validate the model’s efficacy.

The journey begins with a thorough analysis of the dataset comprising images from the three planets. Initial efforts focus on preprocessing techniques, including normalization and the potential transformation of images from RGB to grayscale, aimed at optimizing the data for better model performance. The project leans on the versatility of CNNs for feature extraction, tapping into their prowess in identifying patterns and features crucial for accurate classification.

As the project advances, the Random Forest classifier is brought into play, chosen for its robustness and the ensemble approach that significantly reduces the risk of overfitting by aggregating predictions from multiple decision trees. The project delves into hyperparameter tuning for both CNNs and the Random Forest model, aiming to strike a balance between model complexity and generalization capability. Special attention is devoted to parameters such as the depth of trees, the number of estimators, and the CNN architecture to enhance model performance.

One of the pivotal challenges tackled is the model’s ability to generalize across different planetary environments. Strategies like training on a diversified dataset encompassing images from Earth and Mars and testing on Saturn are employed to evaluate and enhance the model's adaptability to unseen data. Moreover, the reduction of feature space, derived as a counterintuitive yet effective approach, underscores the importance of feature selection in improving classification outcomes.

The project culminates in a comprehensive evaluation across all three planets, demonstrating the model's robustness and adaptability. Despite facing challenges in achieving optimal performance on Neptune, the project sheds light on alternative approaches and underscores the significance of continuous improvement and adaptation in machine learning projects.

In summary, this project illustrates the confluence of CNNs and Random Forest classifiers in tackling the complex challenge of planetary surface classification. Through a meticulous process of training, evaluation, and refinement, it demonstrates the potential of machine learning in astronomical applications, paving the way for future explorations in interplanetary classification tasks.

## Files
- **NNcar - Project files**: Directory containing all the necessary files to run the project
- **NNcar - Animation.gif**: Animation of a possible test on the Earth environment
