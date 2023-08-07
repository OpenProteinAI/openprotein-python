Protein Language Models & Embeddings
=======================================

OpenProtein's Python client provides access to our powerful Embeddings API, designed to generate high-quality protein sequence embeddings. These embeddings are derived from a range of proprietary and open-source models, enabling users to select the most suitable tool for their needs.

The Embeddings Workflow presents a variety of models including:

1. Prot-seq: A proprietary model delivering high-performance protein sequence embeddings.
2. Rotaprot-large-uniref50w: This proprietary model is specifically trained for robust inference capabilities.
3. Rotaprot-large-uniref90-ft: This model is a fine-tuned version of rotaprot-large-uniref50w.
4. ESM1 Models: These models are community-based offerings that use the ESM1 language model as their basis.
5. ESM2 Models: Also community-based, these models use the ESM2 language model.

Each model has its unique characteristics, such as number of parameters, maximum sequence length, dimension, and supported output types, allowing you to customize your workflow to your specific requirements.

You can see details of each of these models (and more) with our python client. Read on to learn more. 

.. note::
   For a practical example of using the this workflow, see the :doc:`embedding workflow notebook <demos/embedding_demo>`.


Index
--------------------
.. toctree::
   :maxdepth: 2

   embedding
   demos/embedding_demo.ipynb
