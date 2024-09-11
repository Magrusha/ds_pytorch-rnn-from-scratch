# Recurrent Neural Nets

Summary: This project is an introduction to recurrent neural nets (RNN) and their evolution to LSTM and GRU

💡 [Нажми сюда](https://new.oprosso.net/p/4cb31ec3f47a4596bc758ea1861fb624), **чтобы поделиться с нами обратной связью на этот проект**. Это анонимно и поможет нашей команде сделать обучение лучше. Рекомендуем заполнить опрос сразу после выполнения проекта.

## Contents

1. [Chapter I. Preamble](#chapter-i-preamble)
2. [Chapter II. Introduction](#chapter-ii-introduction) \
    2.1. [RNN block mechanism](#rnn-block-mechanism) \
    2.2. [Output of RNN](#output-of-rnn) \
    2.3. [Problems of vanilla RNN](#problems-of-vanilla-rnn) \
    2.4. [Embedding layer](#embedding-layer)
3. [Chapter III. Goal](#chapter-iii-goal)
4. [Chapter IV. Instructions](#chapter-iv-instructions)
5. [Chapter V. Task](#chapter-v-task)
6. [Chapter VI. Bonus part](#chapter-vi-bonus-part)

## Chapter I. Preamble

We have already studied some of the main machine learning models, and in all cases, we fed an x vector as input, which describes the characteristics of the data sample we observed. However, what if the observed sample is characterized by a sequence of events? For example, we want to predict an engine breakage based on various sensors, which produce records every 10 seconds. The standard approach is to build the same vector x based on the different aggregations of the sequences, such as taking an average or looking at how many times some sensor's value was above the norm. But since we have neural networks that are famous for their flexibility, can we use them for this task?

## Chapter II. Introduction

### RNN block mechanism

When the data in the problem under study is represented as sequences, then fully connected neural networks are poorly suited for learning, since we feed vectors of a fixed length to the input and output. The advantage of RNN is the ability to process sequences.

This is achieved through the following mechanisms:

- each element of the sequence $`x_t`$ is processed one by one using the so-called RNN block
- the RNN block takes 2 vectors as input: the sequence element vector $`x_t`$ and the state vector $`h_(t-1)`$. This block returns 1 vector in total - state vector $`h_t`$
- the most important - the RNN output of block $`h_t`$ is used as one of the inputs for the next element of the sequence (see figure below)
- for the first element input state vector $`h_0`$ initialized with zeros

![RNN block](misc/images/rnn_block.png)

Thus, when processing an element of the sequence RNN save the information it needs in the $`h_t`$ vector and passes it on. This is what allows the neural network to learn the connections between the elements of the sequence. For example, to learn the relationship between the gender of a noun and the adjective related to it. And this is what is called the memory of RNN.

### Output of RNN

The output of the entire Recurrent Neural Network can be either a sequence or a single value. Depends on what problem we are solving. If we are translating text, then we must get a sequence of words in another language. And if we classify the letter as spam or not spam, then we must translate the input sequence into a probability. The possible variations are illustrated below:

![Output of RNN](misc/images/output_rnn.jpg)

### Problems of vanilla RNN

However, the classical RNN block has a number of disadvantages. It is difficult for vanilla RNN to learn relationships between elements of a sequence that are far apart. Imagine that once a month you buy a water filter from a supermarket near your home. And between these purchases, you go there once a week, filling bags with dozens of goods. If this sequence of goods is fed to the input of the RNN block, then by the time of the next purchase of the filter, the state vector h_t is unlikely to save any trace of the previous one.

The above problem is closely related to the vanishing gradient problem. If the loss function is calculated based on the last element of the sequence, then with backward propagation the gradient can strongly decay and be near-zero for the first elements of the sequence. To understand why this happens, refer to the figure below. The key point for both problems is that during forward and backward paths we multiply on the same matrix many times.

![Problem](misc/images/problem.png)

Чтобы избежать этих проблем было предложено модифицировать структуру RNN блока, добавив внутрь cell state, который способен забывать и обновлять информацию внутри себя, при помощи осторожно регулируемых gate слоев. Из этих соображений и было предложены новые архитектуры, основными из которых стали LSTM и GRU.

Разобраться, как они устроены в деталях и чем различаются - часть вашего задания.

To avoid these problems, it was proposed to modify the structure of the RNN block by adding a cell state inside, which is able to forget and update information within itself, using carefully regulated gated layers. From these considerations, there were proposed several new architectures, but the main ones are LSTM and GRU. Understanding how they work in detail and how they differ is part of your task. You can use [these materials](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) to dive deeper if you wish.

### Embedding layer

Допустим у нас есть последовательность цен для акции фиксированной компании. Если мы будем решать задачу прогнозирования цены на следующий день, то на вопрос “как сформировать вектор x” ответ найти не сложно. Мы можем заполнить его ценой открытия и закрытия предыдущего дня, скользящими средними, финансовыми показателями и прочим.

Let's say we have a sequence of some fixed company stock prices. If we want to solve the problem of predicting the price for the next day, then it is not difficult to find an answer to the question “how to form a vector x”. We can fill it in with the previous day's open and close price, moving averages, financials records, and more. And then, combining these values into the vector for each day over the past year, we get the sequence we need.

Но что делать, если у нас последовательность состоит не из непрерывных величин, а из символов или дискретных значений (например идентификатор купленного товара в магазине). В таком случае обычно используют embedding слой, который попросту каждому символу/слову/идентификатору/прочему ставит в соответствие вектор фиксированного размера. Но самое главное, что все значения внутри этих векторов - обучаемы, т.е. в процессе тренировки нейронной сети они обновляются при помощи градиентного спуска, также как и веса линейных слоев. При этом, если нам удалось достаточно хорошо обучить нашу нейронную сеть, эмбеддинги токенов (так обычно обозначают набор возможных символов/слов/идентификаторов/прочего) могут наследовать хорошо известные нам свойства. Классический пример, иллюстрирующий этот принцип, это наследования отношения 2 слов: слово “король”, соотносится к слову “мужчина”, также как слово “королева” к слову “женщина”. Ничто не мешает записать это выражение в векторном виде:

But what if our sequence does not consist of continuous values, but of symbols or discrete values (for example, the identifier of a purchased product in a store). In this case, an embedding layer is usually used, which simply assigns a fixed size vector to each character / word / identifier / other. And the most important thing is that all these values inside the vector are trainable, i.e. during the training of the neural network, they are updated using gradient descent, as well as the weights of the linear layers. If we manage to train our neural network well enough, embeddings of the tokens (set of possible characters/words/identifiers/etc.) can inherit properties that are well known to us. A classic example illustrating this principle is the inheritance of a 2-word relationship: the word "king" is related to the word "man" in the same way as the word "queen" is related to the word "woman". Nothing prevents us from writing this expression in vector form:

vector(King) - vector(Man) = vector(Queen) - vector(Woman)

![Vectors](misc/images/vectors.png)

## Chapter III. Goal

The goal of this task is to get a deep understanding of how to build models based on RNN, GRU and LSTM architectures.

## Chapter IV. Instructions

- This project will only be evaluated by humans. You are free to organize and name
your files as you desire.
- Here and further we use Python 3 as the only correct version of Python.
- For training deep learning algorithms you can try [Google Colab](https://colab.research.google.com/). It offers kernels
(Runtime) with GPU for free that is faster than CPU for such tasks.
- The norm is not applied to this project. Nevertheless, you are asked to be clear
and structured in the conception of your source code.
- Store the datasets in the subfolder data

## Chapter V. Task

In this project, you will work with a dataset of names. We will represent any name as the sequence of letters and will try to train different models for name generation and name classification.

You can use any framework that you find convenient: PyTorch, TensorFlow, Keras,
etc.

### Dataset

You will work with the open dataset of baby names - [https://data.world/alexandra/baby-names](https://data.world/alexandra/baby-names). The dataset consists of 6782 records. Every record contains the baby name and gender.

### Task

1. Introduction \
    a. Describe for what types of task recurrent neural networks are useful for \
    b. Provide 2 examples for one-to-many, many-to-many, many-to-one formulation of the problem \
    c. Imagine that you have RNN that classifies the language of incoming words. Illustrate (in paint, drawio, etc.) with blocks how this model will process the word “Hello”. Draw every input and output. Define every block with its parameters. Illustrate what will be the initial value of the hidden state. \
    a. Explain the main difference between vanilla RNN, LSTM and GRU. Draw LSTM or GRU similar way to RNN
2. Data Analysis \
    a. Download dataset from the link above or from the course files \
    b. Plot histogram of names length \
    c. Check if the dataset balanced or not in term of gender \
    d. Count frequency of every letter
3. Data preparation \
    a. Encode gender \
    b. Transform names to lowercase \
    a. Extend every name with ‘start of name’ prefix (e.g. `<SOS>`), ‘end of name’ postfix (e.g. `<EOS>`) and pad the name with ‘space’ (e.g. ‘ ’) up to maximal length \
    c. Create the array with all tokens (letters and special symbols) that will be used to encode every name. Create mapping from token to integer id and vice versa \
    d. Present every name as the sequence of integers using mapping from b. Every sequence should have the same length \
    a. Split dataset to train, valid and test part (take 80-10-10%). Be sure to use random_state
4. Recurrent Neural Network \
    a. Implement network that contains: \
        i. embedding layer \
        ii. vanilla recurrent neural network block (do not use already implemented blocks from \ libraries, expect that you make this block from scratch) \
        iii. linear layer to transform the output of RNN block to logits of next tokens output \
        iv. linear layer to transform the last token output of RNN block to logit of gender \
        v. (you can also add dropout layer if you wish) \
        vi. sigmoid to transform logit of gender class into probability \
        vii. don’t forget to implement method for initialization of hidden sate \
    b. Train the network using only loss for next letter prediction: \
        i. take the output from softmax layer for every token in sequence except the last \
        ii. take the next token as the label \
        iii. compute Cross Entropy loss and perform backward step \
    c. Implement method to generate names from trained model \
        i. method should be stochastic - i.e. next letter should be sampled using probabilities based on logit layer \
        ii. provide the parameter to control start of the name \
        iii. provide the parameter to control stochasticity of the next letter. Usually this parameter is called temperature (comes from physics). Higher temperature produces more chaotic outputs, smaller temperature converges to the single most likely output \
        probs = softmax(logits / temperature, dim=-1)
    d. Evaluation \
        i. plot loss convergence on train and valid data \
        i. sample 10 names. Are they realistic? (Do not forget to turn the model into eval mode) \
        ii. predict gender of names from test dataset and calculate ROC AUC score
5. Gated Recurrent Unit \
    a. Repeat steps 4.a, 4.b and 4.d but with GRU model as recurrent neural network block
6. LSTM \
    a. Repeat steps 4.a, 4.b and 4.d but with LSTM model as recurrent neural network block
7. odels comparison \
    a. Suggest your own metric that can be used for name generator models. Please, try to answer this question before you read the next one \
    b. Explain what ytixelprep measure (name of the metric is reverted to avoid spoiler) \
    c. Derive how does this metric is connected with Cross Entropy loss \
    d. Implement 2 versions of this metric (with and without Cross Entropy loss) \
    e. Derive the value of this metric for a “random” model that will return equal probability for every letter. Calculate the value of this metric in the notebook \
    f. Calculate the value of this metric for a “most popular letters” model that will return frequency of letters (frequency of all letters should sum up to 1). \
    g. Compare the models
8. Gender classification \
    a. Let's train RNN, GRU and LSTM but with another loss function \
        i. take the probability of gender class \
        ii. take the name gender as the label \
        iii. compute loss and perform backward step \
        iv. try Binary Cross Entropy loss and Negative Log Likelihood loss \
        v. explain what loss is better and why \
    b. Evaluation \
        i. plot loss convergence on train and valid data \
        i. sample 10 names. Are they realistic? (Do not forget to turn the model into eval mode) \
        ii. predict gender of names from test dataset and calculate ROC AUC score
9. Vanishing gradient \
    a. Take subset of training dataset where names length is equal to its mode
    b. Implement collection of gradients during model fitting for embedding layer. The collected tensor should have the shape (n_epochs, max_name_length, n_tokens, embedding_size). \
        i. in pytorch you can use register_backward_hook \
        ii. in tensorflow you can use tf.GradientTape() or custom Callback \
    c. Debug the gradient collection using batch_size=1 and n_epochs=1 \
        i. calculate the norm of every vector along the last dimension. You will got tensor with shape (1, max_name_length, n_tokens) \
        ii. compare which tokens have non-zero gradient norm with the tokens of the names
        (pay attention that register_backward_hook will collect the gradient going from last token to the first one) \
    d. Calculate the Frobenius norm of gradient matrix for every epoch and token position. The resulted tensor should have the shape (n_epochs, max_name_length) \
    e. Take the mean over epochs \
    f. Train models for gender classification task. Plot and compare the distribution of mean gradient norm over token position for RNN, GRU and LSTM architectures. \
    g. Repeat steps b-f when training the model for the next letter prediction task. What is the difference? \
    h. Explain why we took the subset of the training dataset with fixed length names in the step a \
10. Multitask learning \
    a. Train GRU or LSTM model with 2 loss functions at once: first is the loss from predicting next letter, second is the loss from gender classification \
    b. Evaluate \
        i. plot loss convergence for both loss functions on train and valid data \
        ii. sample 10 names. Are they realistic? (Do not forget to turn the model into eval mode) \
        iii. predict gender of names from test dataset and calculate ROC AUC score
11. Visualizing embeddings \
    a. Get embeddings of every letter from the model with multitask loss \
    b. Use TSNE model to compress vectors into 2 dimensions \
    c. Use PCA model to compress vectors into 2 dimensions \
    d. Split letters to vowels and consonants \
    e. Use scatterplot to visualize letter position in 2 dimensions for both compressions \
    f. What can you conclude from the plots?

### Submission

Save your model from multitask learning in the pickle format. Your peer will load it and use it to make predictions for the test dataset once again (don’t forget to use the same random state). The predictions should be stored in a file with the name test_predictions.csv.

You need to achieve roc_auc_score at least equal to 0.85 on the test dataset.

Your repository should contain one or several notebooks with your solutions.

## Chapter VI. Bonus part

- Try to achieve an even better performance of gender classification on the test dataset – ROC AUC 0.92+
- Suggest the method to get the embedding of every name. Collect them, compress and plot using the split by gender
- Train gradient boosting trees for gender classification task. What features will you use? Evaluate performance on test dataset and compare.
- Train torch/tensorflow/keras/etc built-in RNN, LSTM, GRU models and compare their performance with your models

>Please leave feedback on the project in the [feedback form.](https://forms.yandex.ru/cloud/646b47bce010db2a89f1f5e6/) 
