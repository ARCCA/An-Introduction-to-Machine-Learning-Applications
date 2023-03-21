---
title: "What is Machine Learning"
teaching: 45
exercises: 0
questions:
- "The basics."
objectives:
- "Understand the differences between artificial intelligence, machine learning and deep learning."
- "Familiarize with some of the most common ways to classify machine learning algorithms"
keypoints:
- Machine Learning is the science of getting computers to learn, without being explicitly programmed.
- "Machine Learning main objective is to find new useful representation that help understand hidden patterns in data."
- There are several ways to classify machine learning algorithms based on the problem they are trying to solve, how is data processed and how much human supervision is required.
---

## Differences between Artificial Intelligence, Machine Learning and Deep Learning.
The best way to try to understand the differences between them is to visualize them 
as concentric circles with Artificial Intelligence (the originator field) as the 
largest, then Machine Learning (which flourished later) fitting as a subfield of AI,
and finally Deep Learning (the driving force of modern AI explosion) fitting inside
both.

<img src="{{ page.root }}/fig/ai-ml-dl-relations.svg" alt="AI, ML and DL relationship" width="25%" height="25%" />

### Artificial Intelligence
[Ada Lovelace](https://writings.stephenwolfram.com/2015/12/untangling-the-tale-of-ada-lovelace/) is considered as the first computer programmer in history and while she 
was the first to recognise the enormous potential of the Analytical Engine and 
universal computing, she also noted what she considered as an intrinsic limitation

> *“The Analytical Engine has no pretensions whatever to originate anything. It can
>   do whatever we know how to order it to perform.... Its province is to assist us 
>   in making available what we’re already acquainted with.”*

For several decades this remained as the default view on the topic. However the 
computing developments reached by the 1950s where significant enough to bring Alan 
Turing to pose the question [*"Can machines think?"*](https://www.csee.umbc.edu/courses/471/papers/turing.pdf).
This sprang the development of the new scientific field of AI that could be defined 
as: 

> *The effort to automate intellectual tasks normally performed by humans*.

In this way, AI is a general field that encompasses Machine Learning and Deep 
Learning, but that also includes many more approaches that don’t involve any 
learning. For a fairly long time, many experts believed that human-level AI could be
achieved by having programmers handcraft a sufficiently large set of explicit rules 
for manipulating knowledge. This approach is known as **symbolic AI** (it represents
a problem using symbols and then uses logic to search for solutions), and it was the 
dominant paradigm in AI from the 1950s to the late 1980s. It reached its peak 
popularity during the expert systems boom of the 1980s. 
<!--
For several years a lack of computing power prevented researchers from making significant progress in the development of Artificial Intelligence. Only big universities and private companies could afford the expensive computers with the campabilities necessary to experiment in the field.
-->

<!--
https://www.forbes.com/sites/gilpress/2020/04/27/12-ai-milestones-4-mycin-an-expert-system-for-infectious-disease-therapy/?sh=57eb49f476e5
-->

#### Expert systems

{::options parse_block_html="true" /}
<div>
<ul class="nav nav-tabs" role="tablist">
  <li role="presentation" class="active">
   <a data-es-name="inco" 
      href="#INCO" 
      aria-controls="INCO" 
      role="tab" data-toggle="tab">INCO</a>
  </li>
  <li role="presentation">
   <a data-es-name="mycin"
      href="#MYCIN"
      aria-controls="MYCIN"
      role="tab" 
      data-toggle="tab">MYCIN</a>
  </li>
  <li role="presentation">
   <a data-es-name="dendral" 
      href="#Dendral" 
      aria-controls="Dendral" 
      role="tab" 
      data-toggle="tab">Dendral</a>
  </li>
</ul>

<div class="tab-content">

<article role="tabpanel" class="tab-pane active" id="INCO">
The [**INCO**](https://ntrs.nasa.gov/citations/19960001823) (Integrated Communications
 Officer) Expert System Project (IESP) was undertaken in 1987 by the Mission
Operations Directorate (MOD) at NASA's Johnson Space Center (JSC) to explore the use 
of advanced automation in the mission operations arena.
<img src="{{ page.root }}/fig/stockvault-discovery-space-shuttle208103.jpg"
     alt="Space Shuttle - INCO"
     width="25%" />
</article>

<article role="tabpanel" class="tab-pane" id="MYCIN">
[**MYCIN**](https://en.wikipedia.org/wiki/Mycin) was an early backward chaining 
expert system that used artificial intelligence to identify bacteria causing severe 
infections - MYCIN operated using a fairly simple inference engine and a knowledge 
base of ~600 rules. -  MYCIN was developed over five or six years in the early 1970s
at Stanford University. It was written in Lisp 
<img src="{{ page.root }}/fig/MYCIN.jpg"
     alt="MYCIN"
     width="20%"
     height="20%" />
</article>

<article role="tabpanel" class="tab-pane" id="Dendral">
[**Dendral**](https://en.wikipedia.org/wiki/Dendral) was an artificial intelligence 
project of the 1960s for the specific task of helping organic chemists in identifying
unknown organic molecules by analyzing their mass spectra and using knowledge of 
chemistry. This software is considered the first expert system because it automated 
the decision-making process and problem-solving behavior of organic chemists.
<img src="{{ page.root }}/fig/Caffeine_structure.svg"
     alt="Caffeine Structure"
     width="20%"
     height="20%" />
</article>

</div>
</div>

[anaconda]: https://www.anaconda.com/
[jupyter]: https://jupyter.org/
[python]: https://www.python.org/

<!-- This is commented out.
Deep learning isn’t always the right tool for the job—sometimes there isn’t enough data for deep learning to be applicable, and sometimes the problem is better solved by a different algorithm.

 - Probabilistic modelling is the application of the principles of statistics to data analysis.
   - **Naive Bayes** (Multinomial Naive Bayes Classifier) - ais a type of machine-learning classifier based on applying Bayes’ theoremi all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature (a strong, or “naive” assumption, which is where the name comes from), given the class variable. For example an email maybe considered to be spam if it contains a set of words, regardless of the order of these words. They require a small amount of training data to estimate the necessary parameters. It has a high bias since it ignores relationships between words but because it works well in practice, it is said to have low variance
   - **Logistic regression** predicts whether something is true or false instead of predicting something continues like size (small vs big mice). Also, instead of fitting a line to the data, logistic regression fits an "S" shaped "logistic function". The curve goes from 0 to 1 and it means that the curve tells you the probability of the mouse being small based on its weight. Although logistic regression tells the probability that a mouse is big or not it is usually used for classification. For example if the probabiity of a mouse being small is less than 50% then it is classified as big. Logistic regression can work with continuous data as well as with discrete data. Logistic regression's ability to provide probabilities and classify new samples using continuous and discrete measurements makes it a popular machine learning method. One big difference between linear regression and logistic regression is how the line is fit to the data, with linear regression we fit the line using *least-squares* while with logistic regression we use *maximum likelihood*
 - Early neural networks
 - Kernel methods
 - Decision trees, random forests
 - Modern neural networks
-->

Although symbolic AI proved suitable to solve well-defined, logical problems, such as
playing chess, it turned out to be intractable to figure out explicit rules for 
solving more complex, fuzzy problems, such as image classification, speech 
recognition, and language translation. A new approach arose to take symbolic AI’s 
place: Machine Learning.

## Machine Learning
Machine learning arises from the question: could a computer go beyond *“what we know 
how to order it to perform” and learn on its own how to perform a specified task?* 
Could a computer surprise us? Rather than programmers crafting data-processing rules 
by hand, could a computer automatically learn these rules by looking at data?

This question opened the door to a new programming paradigm. In classical programming
, the paradigm of symbolic AI, humans input rules (a program) and data to be 
processed according to these rules, and out come answers. With Machine Learning, 
humans input data as well as the answers expected from the data, and out come the 
rules. These rules can then be applied to new data to produce original answers.

<img src="{{ page.root }}/fig/ml-programming-paradigm.svg"
     alt="AI, ML and DL relationship"
     width="40%"
     height="40%" />

Using data to predict something can be categorized as Machine Learning, a very simple
example of this is to develop a linear regression model based, for example, on several
mice weight and size measurements and use it predict the size of a new mouse based on
its weight. In general, to do Machine Learning, we need three things:

 - Input data points: for instance, if the task is speech recognition, these data 
   points could be sound files of people speaking. If the task is image tagging, they
   could be pictures. 
 - Examples of the expected output: in a speech-recognition task, these could be 
   human-generated transcripts of sound files. In an image task, expected outputs 
   could be tags such as “dog,” “cat,” and so on. 
 - A way to measure whether the algorithm is doing a good job: this is necessary in 
   order to determine the distance between the algorithm’s current output and its 
   expected output. The measurement is used as a feedback signal to adjust the way 
   the algorithm works. This adjustment step is what we call learning.

The central problem in Machine Learning is to *meaningfully transform data*, this is,
to learn useful *representations* of the input data at hand, representations that 
would get us closer to the expected output. A representation is, at its core, a 
different way to look at data—to represent or encode data. Machine Learning models 
are all about finding appropriate representations for their input data 
transformations of the data that make it more amenable to the task at hand, such as a
classification task.

> ## Learning by changing representations
> Consider a number of points distributed in a xy-coordinate system. Some of them are
>  white and some are black.
>
> <img src="{{ page.root }}/fig/example-1-raw-data.svg" 
>      alt="Example 1 - Raw Data" width="20%" height="20%" />
>
> And we are given the task to develop an algorithm to calculate the probability of 
> the point being black or white given its x-y coordinates. In this case,
> - The inputs are the coordinates of our points
> - The expected outputs are the colours of our points
> - The measure of success would be the percentage of points correctly classified
>
> One way to solve the problem is by applying a coordinate change (a new 
> representation of our data). This new representation would allow us to classify our
> points with a more simple set of rules: "Black points are those such that x>0"
>
> In this case, we defined the coordinate change by hand. But if instead we tried 
> systematically searching for different possible coordinate changes, and used as 
> feedback the percentage of points being correctly classified, then we would be 
> doing Machine Learning. *Learning*, in this context, describes an automatic search 
> process for better representations. 
>
>  Coordinate change             |  Better representation
>  :-------------------------:|:-------------------------:
>  <img src="{{ page.root }}/fig/example-1-coordinate-change.svg" alt="Example 1 - Coordinate Change" width="50%" height="50%" /> | <img src="{{ page.root }}/fig/example-1-better-representation.svg" alt="Example 1 - Better Representation" width="50%" height="50%" />
{: .callout}

All Machine Learning algorithms consist of automatically finding such transformations
that turn data into more useful representations for a given task. These operations 
can be coordinate changes, as you just saw, or linear projections, translations, 
nonlinear operations, and so on. Machine Learning algorithms aren’t usually creative
in finding these transformations; they’re merely searching through a predefined set 
of operations, also called a hypothesis space.

With the previous description in mind, we can summarize Machine Learning as:
> *the field of study that gives computers the ability to learn without being 
>  explicitly programmed.*
> —Arthur Samuel, 1959

Or more formally:
> *A computer program is said to learn from experience E with respect to some task T 
>  and some performance measure P, if its performance on T, as measured by P, 
>  improves with experience E.*
> —Tom Mitchell, 1997

A machine-learning system is trained rather than explicitly programmed. It is
presented with many examples relevant to a task, and it finds statistical structure 
in these examples that eventually allows the system to come up with rules for 
automating the task.

<!--
Machine learning is tightly related to mathematical statistics, but it differs from statistics in several important ways. Unlike statistics, machine learning tends to deal with large, complex datasets (such as a dataset of millions of images, each consisting of tens of impractical. As a result, machine learning, and especially deep learning, exhibits comparatively little mathematical theory—maybe too little—and is engineering oriented. It’s a hands-on discipline in which ideas are proven empirically more often than theoretically.
-->

> ## When to use Machine Learning
> Machine Learning methods are great when:
> - the solution of a problem requires lots of parameter tweaking and/or writing
>   long lists of rules.
> - there is no known optimal solution for the problem at hand.
> - the problem requires adapting to constantly changing data.
> - we wish to obtain more insights about complex problems and large amounts of data.
{: .callout}

## Common Machine Learning problems
There are many classes and subclasses of Machine Learning problems based on what the
prediction task looks like. Some of the most common ones include:
 - **Classification** - We want to predict one of several options. The possible 
   outcomes are called classes. The MNIST problem is a 10 classes classification 
   problem.
 - **Regression** - We want to predict a continuous value. For example, mice weight 
   given its size.
 - **Clustering**: The goal is to split up the data in such a way that points within
   a single cluster are very similar and points in different clusters are different.
 - **Association rule learning** - Helpful to infer likely association patterns in 
   data.
 - **Structured output** - Useful to create complex output.
 - **Ranking** - The goal is to retrieve answers to a particular query, ordered by 
   their relevance.
 - **Recommenders**: Provide suggestions to users based on their preferences.

## Types of Machine learning
Although there is no formal Machine Learning classification system yet, they are 
commonly described and classified in broad categories based on:

- How much human supervision is required. There are four major categories:
  - **Supervised learning**: this method requires feeding the algorithm with "right"
    answers or solutions, also known as labels. Some of the most important supervised
    learning algorithms: k-Nearest Neighbours, Linear Regression, Logistic Regression,
     Support Vector Machines (SVMs), Decision Trees and Random Forests, Neural 
    networks.
  - **Unsupervised learning**: in contrast, in this algorithm the system tries to 
    learn without a teacher, this is, with unlabelled data. Some of the most 
    important unsupervised learning algorithms are: Clustering (K-Means, DBSCAN, 
    Hierarchical Cluster Analysis (HCA)), Anomaly detection and novelty detection 
    (One-class SVM, Isolation Forest), Visualization and dimensionality reduction 
    (Principal Component Analysis (PCA), Kernel PCA, Locally-Linear Embedding (LLE),
    t-distributed Stochastic Neighbour Embedding (t-SNE)), Association rule learning
    (Apriori, Eclat).
  - **Semisupervised learning**: these algorithms are able to work with some 
    partially labeled training data, typically lots of unlabelled data and a few bits
    of labelled data. Most semisupervised learning algorithms are combinations of 
    unsupervised and supervised algorithms.
  - **Reinforcement Learning**
- Whether the model can learn incrementally or requires reprocessing entire datasets.
  There are two major categories:
  - **Batch learning**: This is also known as offline learning. In here the system is
    trained using all available data (which typically is both time and 
    computationally expensive). Once the system is trained, it is launched in 
    production runs where it applies what it learned but without adding any new 
    knowledge.
  - **Online (incremental) learning**: also known as online learning. In here the 
    system can be trained incrementally by feeding it small data batches. This type 
    of learning process is more useful where limited computational resources are 
    available (e.g. when a huge database cannot be entirely contained in the 
    machine's memory).
- Whether the model is able to find new patterns in the data or relies on comparing 
  new data points to known data points.
  - **Instance-based learning**: the system relies completely on the already learned
    examples and a similarity measure to generalize to new observed cases. Not the 
    most sophisticated method, but can produce good results.
  - **Model-based learning**: the system builds a model using a set of examples and 
    then use it to make predictions. Depending on the model complexity there will 
    typically be a number of parameters that need to be tuned by the system (using 
    some kind of performance measurement) to find a set that makes the model perform
    best.

## Main challenges

- **Insufficient Quantity of Training Data**: Machine Learning algorithms required 
  high amounts of data to work properly, typically in the order of thousands of 
  examples even for fairly simple problems and of millions for more complex problems
  (image or speech recognition).
- **Nonrepresentative Training Data**: The predictions made by a Machine Learning 
  model depend not only on the amount of data provided to the algorithm but also on 
  its significance. If the data is not representative of the samples the model will 
  encounter in real life, the predictions cannot be accurate either.
- **Poor-Quality Data**: And related with the previous point, if the data contains a 
  high number of errors (caused for example by noise, incomplete samples, poor 
  measurement equipment), the algorithm will struggle to find patterns in the data.
- **Irrelevant Features**: In the other hand, providing more data (even high-quality 
  data) does not guarantee a more accurate model, especially if the additional data 
  describes unimportant aspects of the samples.
- **Overfitting the Training Data**: This can happen when using a relatively complex 
  model that might be able to describe better the training data set but fails when 
  exposed to real samples. A typical solution is to reduce the model complexity.
- **Underfitting the Training Data**: The opposite of creating an overfitting model 
  is creating an underfitting one and this typically happens when the model is too 
  simple to learn the underlying structure of the data.

<!-- This is commented out.
> ## Why *deep* learning
> Deep learning is a specific subfield of machine learning: a new take on learning representations from data that puts an emphasis on learning successive layers of increasingly meaningful representations. The deep in deep learning isn’t a reference to any kind of deeper understanding achieved by the approach; rather, it stands for this idea of successive layers of representations. How many layers contribute to a model of the data is called the depth of the model.
> Modern deep learning often involves tens or even hundreds of successive layers of representations—and they’re all learned automatically from exposure to training data.
> For our purposes, deep learning is a mathematical framework for learning representations from data.
{: .callout}

## Before deep learning: a brief history of machine learning.

## Why deep learning? Why now?
-->

## Computational frameworks
Deep learning frameworks allow a user to define networks either via a config (like 
Caffe) or programmatically like Theano, Tensorflow, or Torch. Furthermore, the 
programming language exposed to define networks might vary, like Python in the case 
of Theano and Tensorflow or Lua in the case of Torch. An additional variation on them
is whether the framework provides define-compile-execute semantics or dynamic 
semantics (as in the case of PyTorch). 

 - [**Sci-Kit Learn**](https://scikit-learn.org/stable/): is an open source project,
   meaning that it is free to use and distribute, and anyone can easily obtain the 
   source code to see what is going on behind the scenes. The scikit-learn project is
   constantly being developed and improved, and it has a very active user community. 
   It contains a number of state-of-the-art machine learning algorithms, as well as
   comprehensive documentation about each algorithm. scikit-learn is a very popular 
   tool, and the most prominent Python library for machine learning. It is widely 
   used in industry and academia, and a wealth of tutorials and code snippets are 
   available online. 
 - [**PyTorch**](https://pytorch.org/): is an open source machine learning library 
   based on the Torch library, used for applications such as computer vision and 
   natural language processing, primarily developed by Facebook's AI Research lab 
   (FAIR). is an optimized tensor library for deep learning using GPUs and CPUs. 
   PyTorch is very well suited for research purposes as it makes developing and 
   experimenting with new deep learning architectures relatively easy.
 - [**Tensorflow**](https://www.tensorflow.org/): allows users to define mathematical
   functions via computational graphs and to compute their gradients. Tensorflow is 
   conceptually similar to Theano, and Keras uses both of them as back ends. it has 
   strong backing from industry leaders such as Google. At its essence, Tensorflow 
   allows users to define mathematical functions on tensors (hence the name) using 
   computational graphs and to compute their gradients.
 - [**Keras**](https://keras.io/): is a library that provides highly powerful and 
   abstract building blocks to build Deep and Machine Learning systems. The building 
   blocks Keras provides are built using Theano as well as TensorFlow. Keras supports
   both CPU and GPU computation and is a great tool for quickly prototyping ideas.
 - [**CNTK**](https://docs.microsoft.com/en-us/cognitive-toolkit/): is an open-source
   toolkit for commercial-grade distributed deep learning. It describes neural 
   networks as a series of computational steps via a directed graph. CNTK allows the 
   user to easily realize and combine popular model types such as feed-forward DNNs, 
   convolutional neural networks (CNNs) and recurrent neural networks (RNNs/LSTMs).
 - [**Theano**](https://github.com/Theano/Theano): is a Python library for defining 
   mathematical functions (operating over vectors and matrices), and computing the 
   gradients of these functions. Theano allows the user to define mathematical 
   expressions that encode loss functions and, once these are defined, Theano allows 
   the user to compute the gradients of these expressions.
 - [**Caffe**](https://caffe.berkeleyvision.org/):  a deep learning framework made 
   with expression, speed, and modularity in mind. It is developed by Berkeley AI
   Research (BAIR) and by community contributors.

<!--
 - Amazon Machine Learning
 - Google Cloud ML Engine
Helpful links
https://developers.google.com/machine-learning/crash-course
https://www.coursera.org/learn/machine-learning/
-->

## References
Some further reading to learn more about machine and deep learning.
- Aurélien Géron (2019) *Hands-On Machine Learning with Scikit-Learn, Keras, and 
  Tensorflow Concepts, Tools, and Techniques to Build Intelligent Systems*. 2nd ed.
  US: O'Reilly.
- Andreas C. Müller & Sarah Guido (2016) *Introduction to Machine Learning with 
  Python A Guide for Data Scientists*. 1st ed. US: O'Reilly.
- Francois Chollet (2018) *Deep Learning with Python*. 1st ed. New York: Manning 
  Publications. 
- Delip Rao & Brian McMahan (2019)  *Natural Language Processing with PyTorch Build 
  Intelligent Language Applications Using Deep Learning*. 1st ed. US: O'Reilly.
- Ian Pointer (2019) *Programming PyTorch for Deep Learning*. 1st ed. US: O'Reilly.

## Further training
Some other sources with training material for DL and ML applications:

- [TensorFlow - Learn Machine Learning](https://www.tensorflow.org/resources/learn-ml)
- [Scikit Learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)
- [Coursera - DeepLearning.AI TensorFlow Developer Professional Certificate](https://www.coursera.org/professional-certificates/tensorflow-in-practice)
- [Coursera - Machine Learning](https://www.coursera.org/learn/machine-learning?)
- [Udacity - Intro to TensorFlow for Deep Learning](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187)
- [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)

- [Yann LeCun’s Deep Learning Course at CDS]( https://cds.nyu.edu/deep-learning/)
- [MIT Courseware Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/)
- [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [CALTECH - Learning From Data - Machine Learning Course](https://home.work.caltech.edu/telecourse.html)
- [UC Berkeley - Full Stack Deep Learning](https://fullstackdeeplearning.com/spring2021/)
- [Stanford University - Introduction to Robotics](https://see.stanford.edu/Course/CS223A) 
- [Linear Algebra Review](https://www.cs.cmu.edu/~zkolter/course/linalg/)
- [Carnegie Melon University - Math Background for Machine Learning](https://canvas.cmu.edu/courses/603/assignments/syllabus)
