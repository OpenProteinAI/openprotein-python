=============================================
Welcome to OpenProtein's documentation!
=============================================

.. image:: https://badge.fury.io/py/openprotein-python.svg
    :target: https://pypi.org/project/openprotein-python/
    :align: right
.. image:: coverage.svg
    :target: https://pypi.org/project/openprotein-python/
    :align: right
.. image:: https://anaconda.org/openprotein/openprotein_python/badges/version.svg
    :target: https://anaconda.org/openprotein/openprotein_python
    :align: right

Welcome to OpenProtein.AI! 

OpenProtein.AI is a powerful platform that seamlessly integrates state-of-the-art machine learning and generative models into protein engineering workflows. 

We help you to design better proteins faster by giving you access to cutting-edge protein language models, prediction and design algorithms, as well as data management and model training tools. Easily build and deploy high-performance Bayesian protein function predictors, or apply generative protein language models to design sequence libraries, all via our integrated platform. OpenProtein.ai can be accessed via web App, API and this python client, making it great for both biologists and protein engineers, as well as data scientists and software engineers. 

The documentation is divided into workflows below. For each workflow you can read the docs and see a demo of usage. 

Table of Contents
-----------------

+---+----------------------------------------------------+------------------------------------------------------+
|   | Workflow                                           | Description                                          |
+===+====================================================+======================================================+
| 0 | `Installation`_                                    |Install guide for pip and conda.                      |
|   |                                                    |                                                      |
+---+----------------------------------------------------+------------------------------------------------------+
| 1 | `Session management`_                              |An overview of the OpenProtein Python Client          |
|   |                                                    |& the asynchronous jobs system.                       |
+---+----------------------------------------------------+------------------------------------------------------+
| 2 | `Asssay-based Sequence Learning`_                  |Covers core tasks such as data upload,                |
|   |                                                    |model training & prediction, and sequence design      |
|   |                                                    |                                                      |
+---+----------------------------------------------------+------------------------------------------------------+
| 3 | `De Novo prediction & generative models (PoET)`_   |Covers PoET, a protein LLM for *de novo* scoring,     |
|   |                                                    |as well as sequence generation.                       |
|   |                                                    |                                                      |
+---+----------------------------------------------------+------------------------------------------------------+
| 4 | `Protein Language Models & Embeddings`_            |Covers methods for creating sequence                  |
|   |                                                    |embeddings with proprietary & open-source models      |
+---+----------------------------------------------------+------------------------------------------------------+

.. _installation: installation.html
.. _Session management: overview.html
.. _Asssay-based Sequence Learning: core_workflow.html
.. _De Novo prediction & generative models (PoET): poet_workflow.html
.. _Protein Language Models & Embeddings: embedding_workflow.html


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   installation
   overview
   core_workflow
   poet_workflow
   embedding_workflow
