Asssay-based Sequence Learning
=============================== 

Welcome to the Asssay-based Sequence Learning section of the documentation! Here, we describe how to use our library to perform the core tasks associated with data processing and utilizing the platform's machine learning capabilities.

The Asssay-based Sequence Learning functionality of OpenProtein's Python client library is divided into four main modules: AssayData, Train, Predict, and Design.


.. rubric:: AssayData

Our AssayData module allows you to upload your dataset to OpenProtein's engineering platform. This dataset forms the basis for training, predicting, and evaluating tasks. Your data should be formatted as a 2-column CSV, including the full sequence of each variant and one or more columns for your measured properties.

See the :doc:`AssayData documentation <data>` for more details.

.. rubric::Train

Our Train module provides functions to train models on your measured properties. This step is essential for enabling predictions for new sequences. These workflows also perform cross-validation on your models to estimate uncertainty.

See the :doc:`Train documentation <train>` for more details.

.. rubric:: Predict

With the Predict module, you can make predictions on arbitrary sequences using your trained OpenProtein models. This includes predictions for single sequences as well as single mutant variants of the sequence.

See the :doc:`Predict documentation <train>` for more details.

.. rubric:: Design

The Design module provides the capability to design new sequences based on your objectives using our genetic algorithm.

See the :doc:`Design documentation <design>` for more details.

Remember that these workflows require you to first upload your datasets using the AssayData module and train your models using the Train module.

.. note::
   For a practical example of using this workflow, see the :doc:`core workflow notebook <demos/core_demo>`.

In addition to this documentation, we offer demos of key workflows and provide demo datasets to help you familiarize yourself with our workflows. Happy learning and exploring!

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   data
   train
   design
   demos/core_demo.ipynb