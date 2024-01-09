# ECE449 Machine Learning Bullet Points
Tony Wang & Ziyuan Chen
- Enhanced by ChatGPT
Jan. 16th, 2023

> Prompt:
Help me clarify the upper ocr result, they are about the final question review, generate your output in bullet point and markdown


### Classical ML

#### Topics Covered:
- **Regression and Classification:** 
  - Discussing methods for predicting continuous outputs (regression) and categorical outputs (classification).
- **Supervised vs Unsupervised Learning:**
  - Differentiating between learning with labeled data (supervised) and without labeled data (unsupervised).
- **Model Optimization Types:**
  - Exploring different approaches to optimize machine learning models.
- **Trade-off Between Bias and Variance:**
  - Understanding the balance between underfitting (high bias) and overfitting (high variance).
- **Overfitting & Underfitting, Setting Hyperparameters:**
  - Recognizing when models overfit or underfit data and how to tune hyperparameters accordingly.
- **Validation Set - Why & How (Held-Out/Cross-Validation):**
  - Discussing the importance of validation sets and methods like held-out validation and cross-validation.

#### Specific Techniques and Concepts:
- **Linear Regression and Fitting, Define MSE Loss:**
  - Understanding linear regression and Mean Squared Error (MSE) as a loss function.
- **Derivative in the Context of Optimization:**
  - Explaining the role of derivatives in finding optimal solutions.
- **Extension to Non-Linear (Fit Polynomial):**
  - Extending linear models to fit non-linear relationships using polynomial features.
- **Matrix Form, Closed Form Solution:**
  - Discussing linear regression in matrix form and its closed form solution.
- **Regularization, Choosing Lambda (λ):**
  - Implementing regularization to prevent overfitting and criteria for selecting the regularization parameter (λ).
- **LASSO (No Closed Form):**
  - Understanding LASSO regression, which includes L1 regularization and typically lacks a closed-form solution.

### Unsupervised Learning

#### Key Concepts:
- **KNN (K-Nearest Neighbors) Instance-Based Learning:**
  - Discussing KNN as a simple, non-parametric method in unsupervised learning.
- **SVM (Support Vector Machine):**
  - Exploring SVM for classification tasks.
- **Objective Function in SVM:**
  - Understanding the goal and formulation of the objective function in SVM.
- **Non-Linear Features in SVM:**
  - Extending SVM to non-linear feature spaces.
- **Extension to Multi-Class Classification:**
  - Discussing methods for adapting binary classifiers like SVM for multi-class problems.

### Perception and Newton Method

#### Topics Covered:
- **Limitations:**
  - Addressing the constraints and shortcomings of perception and Newton's method.
- **Linear Classification and Outline Learning:**
  - Understanding the basics of linear classification.
- **Newton as "Weighted Sum" → Multilayer:**
  - Transitioning from the concept of a weighted sum in Newton's method to multilayer networks.
- **Activation Functions:**
  - Discussing the role of activation functions in neural networks.
- **Chain Rule in Optimization:**
  - Applying the chain rule in the context of optimizing neural networks.

### Neural Networks and Associated Concepts

#### Key Elements:
- **Simple Backpropagation and Computational Graph:**
  - Explaining the fundamentals of backpropagation and the use of computational graphs in neural networks.
- **Convolution Keywords:**
  - Discussing key terms associated with convolutional neural networks (CNNs).
- **Number of Parameters in CNNs:**
  - Understanding how to calculate the number of parameters in CNN layers.
- **Dropout:**
  - Explaining dropout as a technique to prevent overfitting in neural networks.
- **Learning Rate Scheduler (LR Scheduler):**
  - Discussing methods for adjusting the learning rate during training.
- **Contribution of Such Generation of ImageNet:**
  - Understanding the impact of ImageNet in advancing neural network technologies.

#### Specific Topics:
- **Receptive Field:**
  - Discussing the concept of the receptive field in the context of CNNs.
  - **Variations:**
    - **|X| COW:** Exploring specific variations or implementations within CNNs.
    - **Depthwise Convolution:** Understanding depthwise convolutions in CNNs.
    - **Grouped Convolution:** Discussing grouped convolutions in CNN architectures.
- **RNN (Recurrent Neural Networks) & Variations:**
  - Exploring the basics of RNNs and their variations for processing sequential data.
- **Captioning, Projection:**
  - Discussing the application of neural networks in tasks like image captioning and feature projection.




### Modern Machine Learning

#### Topics Covered:
- **LSTM (Long Short-Term Memory):**
  - Understanding the architecture of LSTM and its computational graph.
- **Attention & Transformer:**
  - **Key-Query-Value (KQV):** Understanding the KQV mechanism in attention models.
  - **Multi-Head Attention:** Discussing the concept of multi-head attention in Transformer models.
  - **Positional Encoding:** Explaining how positional information is encoded in Transformer models.
  - **Space Attention:** Understanding the rationale behind space attention.

#### Specific Techniques and Concepts:
- **Graph Neural Networks (GNN):**
  - **Tabular Data Processing:** Discussing GNN's application in processing tabular data.
  - **Learning Node Embeddings:** Understanding how GNNs learn representations of nodes in a graph.
  - **$h = \sigma(W * \sum h/N + Bh)$:** Exploring the formulation of a GNN layer.
  - **GCN (Graph Convolutional Network):** Understanding the basics of GCNs.
  - **Not considering SAGE:** A specific note about excluding the study of SAGE (a type of GNN).

### ML Training

#### Key Elements:
- **Data Preprocessing and Normalization:**
  - Discussing the importance of preprocessing and normalizing data in ML workflows.
- **Loss Check:**
  - Understanding the process of monitoring and analyzing loss during training.
- **Object Detection:**
  - Exploring methodologies and models used in object detection tasks.
- **Mask-RCNN:**
  - Discussing the architecture and application of Mask-RCNN in instance segmentation.
- **Bilinear Interpolation:**
  - Understanding the use of bilinear interpolation in image processing within ML models.
- **Meta Learning:**
  - Exploring the concept of meta learning in machine learning.
- **Self-Supervised Learning:**
  - Discussing the theory and ideas behind self-supervised learning approaches.
- **Adversarial Learning:**
  - Understanding the principles and applications of adversarial learning in ML.
- **GAN (Generative Adversarial Networks):**
  - **Components:** Discussing the key parts of a GAN.
  - **Training Process:** Exploring how GANs are trained.
  - **Minimax Game:** Understanding the minimax game theory behind GANs.
- **Cache to Image Pipeline:**
  - Exploring the process of caching data in image processing pipelines.
- **Cycle-Consistency:**
  - Discussing the concept of cycle-consistency in machine learning models.
- **Domain Adaptation:**
  - Understanding the techniques for adapting models to different domains.
- **Variational Autoencoder (VAE) - Encoder (E), Decoder (D):**
  - Exploring the architecture and principles of variational autoencoders.
- **Loss Function:**
  - Discussing various loss functions used in machine learning models.

These topics encompass a range of advanced concepts in modern machine learning and ML training, covering both theoretical foundations and practical implementations.



### NLP (Natural Language Processing)

#### Topics Covered:
- **N-grams:**
  - Understanding the concept of N-grams in text data.
- **Markov Assumption:**
  - Discussing the Markov assumption in the context of language modeling.
- **Smoothing:**
  - Exploring techniques for smoothing in statistical language models.
- **Problems with Smoothing:**
  - Addressing the issues and challenges in applying smoothing techniques.
- **Disadvantage of One-Hot Encoding:**
  - Discussing the limitations of using one-hot encoding for representing text data.

#### Specific Techniques and Concepts:
- **Word2Vec:**
  - Exploring the Word2Vec model for word embedding.
- **GPT & BERT:**
  - Understanding the architecture and applications of GPT and BERT models in NLP.
- **Naive Bayes Classification:**
  - Discussing the application of Naive Bayes in text classification.
- **Bayesian Rule:**
  - Exploring the application of Bayesian rule in NLP.
- **MAP (Maximum A Posteriori):**
  - Understanding the MAP estimation in the context of Bayesian inference.

### Back to Classical Machine Learning

#### Key Elements:
- **Logistic Regression:**
  - Discussing the fundamentals of logistic regression for binary classification.
- **Gradient Update:**
  - Understanding how gradient descent is used for updating model parameters.
- **Decision Trees:**
  - Exploring the concept and application of decision trees in classification and regression.
- **Entropy:**
  - Discussing the concept of entropy in the context of information theory and decision trees.
- **Overfitting:**
  - Addressing the issue of overfitting in machine learning models.
- **Bagging:**
  - Understanding the technique of bootstrap aggregating (bagging) to improve model accuracy.
- **AdaBoost:**
  - Exploring the AdaBoost algorithm for boosting model performance.
- **Unsupervised Learning:**
  - Discussing the principles of unsupervised learning methodologies.
- **Clustering:**
  - Understanding different clustering techniques and their applications.
- **Theoretical KMeans:**
  - Exploring the theory behind the KMeans clustering algorithm.
- **Hard-Soft Assignment:**
  - Discussing hard and soft assignment methods in clustering.
- **GMM & EM (Gaussian Mixture Models & Expectation-Maximization):**
  - Understanding the application of GMMs and the EM algorithm in clustering.
- **Spatial Clustering:**
  - Exploring clustering techniques specifically for spatial data.
- **Evaluating Clusters:**
  - Discussing methods for evaluating the quality of clustering.
- **PCA & Reconstruction Error:**
  - Understanding Principal Component Analysis (PCA) and how reconstruction error is calculated.
- **Degree Matrix and Laplacian:**
  - Discussing the role of degree matrices and Laplacians in graph theory and machine learning.
- **Active Learning:**
  - Exploring the concept and strategies of active learning.
- **Sample Selection:**
  - Understanding techniques for selecting samples in active learning.
- **Stream and Pool Based Learning:**
  - Discussing stream-based and pool-based strategies in active learning.
- **Reloof Density and Uncertainty & Choosing Between Strategy:**
  - Exploring strategies based on density and uncertainty in active learning.

### Reinforcement Learning

#### Topics Covered:
- **Definition:**
  - Understanding the key components like history, state, environment, agent, reward, observation, etc.
- **Policies:**
  - Discussing different types of policies in reinforcement learning.
- **Markov Decision Process:**
  - Exploring the concept of Markov Decision Processes in the context of RL.
- **Bellman Equation:**
  - Understanding the Bellman equation and its role in RL.
- **TD (Temporal Difference), MC (Monte Carlo):**
  - Discussing Temporal Difference and Monte Carlo methods in RL.
- **R Learning (?):**
  - Possibly discussing a specific type of RL algorithm.
- **Eligibility Trace:**
  - Exploring the concept of eligibility traces in RL.
- **Epsilon-Greedy Exploration:**
  - Discussing the epsilon-greedy strategy for exploration in RL.
- **SARSA (?):**
  - Possibly discussing the SARSA algorithm in RL (State-Action-Reward-State-Action).

These topics cover a broad range of areas in NLP and classical machine learning, as well as an introduction to reinforcement learning, providing a comprehensive overview of key concepts and techniques in these fields.