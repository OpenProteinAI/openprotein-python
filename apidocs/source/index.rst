=============================================
Welcome to OpenProtein's documentation!
=============================================

.. image:: https://badge.fury.io/py/openprotein-python.svg
    :target: https://pypi.org/project/openprotein-python/
    :align: right
.. image:: coverage.svg
    :target: https://pypi.org/project/openprotein-python/
    :align: right
.. image:: https://anaconda.org/openprotein/openprotein-python/badges/version.svg
    :target: https://anaconda.org/openprotein/openprotein-python
    :align: right

Welcome to OpenProtein.AI! 

OpenProtein.AI is a powerful platform that seamlessly integrates state-of-the-art machine learning and generative models into protein engineering workflows. 

We help you to design better proteins faster by giving you access to cutting-edge protein language models, prediction and design algorithms, as well as data management and model training tools. Easily build and deploy high-performance Bayesian protein function predictors, or apply generative protein language models to design sequence libraries, all via our integrated platform. OpenProtein.ai can be accessed via web App, API and this python client, making it great for both biologists and protein engineers, as well as data scientists and software engineers. 

The documentation is divided into workflows below. For each workflow you can read the docs and see a demo of usage. 


Table of Contents
-------------------

+----------------------------------------------------+------------------------------------------------------+
| Workflow                                           | Description                                          |
+====================================================+======================================================+
| `Installation`_                                    |Install guide for pip and conda.                      |
|                                                    |                                                      |
+----------------------------------------------------+------------------------------------------------------+
| `Session management`_                              |An overview of the OpenProtein Python Client          |
|                                                    |& the asynchronous jobs system.                       |
+----------------------------------------------------+------------------------------------------------------+
| `Asssay-based Sequence Learning`_                  |Covers core tasks such as data upload,                |
|                                                    |model training & prediction, and sequence design      |
|                                                    |                                                      |
+----------------------------------------------------+------------------------------------------------------+
| `De Novo prediction & generative models (PoET)`_   |Covers PoET, a protein LLM for *de novo* scoring,     |
|                                                    |as well as sequence generation.                       |
|                                                    |                                                      |
+----------------------------------------------------+------------------------------------------------------+
| `Protein Language Models & Embeddings`_            |Covers methods for creating sequence                  |
|                                                    |embeddings with proprietary & open-source models      |
+----------------------------------------------------+------------------------------------------------------+


.. _installation: installation.html
.. _Session management: overview.html
.. _Asssay-based Sequence Learning: core_workflow.html
.. _De Novo prediction & generative models (PoET): poet_workflow.html
.. _Protein Language Models & Embeddings: embedding_workflow.html


Tutorials
------------

We have a range of tutorials to get you started:

Quick-start Guides
=====================

- :doc:`Sequence-based learning <demos/core_demo>`
- :doc:`Sequence embeddings <demos/embedding_demo>`
- :doc:`Getting started with Poet <demos/poet_demo>`

.. toctree::
   :maxdepth: 2
   :caption: Quick-start Guides
   :hidden:

   demos/core_demo.ipynb
   demos/embedding_demo.ipynb
   demos/poet_demo.ipynb

Case-studies
=============

- :doc:`Designing new Chorismate mutase enzymes with PoET notebook <demos/chorismate>`

.. toctree::
   :maxdepth: 2
   :caption: Case-studies
   :hidden:

   demos/chorismate.ipynb


.. toctree::
   :maxdepth: 2
   :caption: Index

   installation
   overview
   core_workflow
   poet_workflow
   embedding_workflow

