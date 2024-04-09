Embeddings
============

Create embeddings for your protein sequences using open-source and proprietary models!

Note that for PoET Models, you will also need to utilize our :doc:`align <align>`. workflow.

endpoints
-----------
.. autoclass:: openprotein.api.embedding.EmbeddingAPI
   :members:
   :undoc-members:

Models 
------------

.. autoclass:: openprotein.api.embedding.OpenProteinModel
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: openprotein.api.embedding.ESMModel
   :members:
   :inherited-members:

.. autoclass:: openprotein.api.embedding.PoETModel
   :members:
   :inherited-members:

.. autoclass:: openprotein.api.embedding.SVDModel
   :members:
   :inherited-members:

Results
---------

.. autoclass:: openprotein.api.embedding.EmbeddingResultFuture
   :members:
   :inherited-members:

.. autoclass:: openprotein.api.poet.PoetScoreFuture
   :members:
   :inherited-members:

.. autoclass:: openprotein.api.poet.PoetSingleSiteFuture
   :members:
   :inherited-members:

.. autoclass:: openprotein.api.poet.PoetGenerateFuture
   :members:
   :inherited-members:
