Fold
============

Create PDBs of your protein sequences via our folding models!

Note that for AlphaFold2 Models, you will also need to utilize our :doc:`align <align>`. workflow.

endpoints
-----------
.. autoclass:: openprotein.api.fold.FoldAPI
   :members:
   :undoc-members:

Models 
------------

.. autoclass:: openprotein.api.fold.ESMFoldModel
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: openprotein.api.fold.AlphaFold2Model
   :members:
   :inherited-members:


Results
---------

.. autoclass:: openprotein.api.fold.FoldResultFuture
   :members:
   :inherited-members:

