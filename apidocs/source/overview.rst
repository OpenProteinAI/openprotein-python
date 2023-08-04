Session management
======================

OpenProtein login 
------------------

Use your username and password credentials generated at sign-up to authenticate your connection to OpenProtein backend.

See the :doc:`APISession documentation <basics>` for more details.


OpenProtein Job System
-------------------------

The OpenProtein platform operates with an asynchronous framework. When initiating a task using our Python client, the system schedules the job, returning a prompt response with a unique Job ID. This mechanism ensures that tasks requiring longer processing times do not necessitate immediate waiting. 

When you submit a task, such as using the method ::

    session.poet.create_msa()

a Future Class is returned for results tracking and access. You can check a job's status using the ``refresh()`` and ``done()`` methods on this Future Class. If you wish to wait for the results, you can use the ``wait()`` method, or the ``get()`` method if the results are already completed.

In addition, you can resume a workflow using the ``load_job`` function along with the unique job ID obtained during task execution. This method will return a Future Class, allowing you to continue from where you left off.

OpenProtein API session
-------------------------

Executing workflows is acheived with the OpenProtein APISession object (see :py:meth:`openprotein.APISession`) ::

    session = openprotein.connect(username="username", password="password")

You then have access to all the workflows: 

For example ::

    session.data.create()

or ::

    session.poet.create_msa()

.. note::
   For practical examples, see the following notebooks:
   
   - :doc:`Core workflow notebook <demos/core_demo>`
   - :doc:`Poet workflow notebook <demos/poet_demo>`
   - :doc:`Embeddings workflow notebook <demos/embedding_demo>`

.. toctree::
   :maxdepth: 2
   :caption: Index:

   basics