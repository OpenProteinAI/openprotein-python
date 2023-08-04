De Novo prediction & generative models (PoET)
====================================================

PoET is a generative protein language model that allows controllable design of protein sequences and variant effect prediction. This model is controlled by providing it with a prompt, a set of sequences that represent homologues, family members, or some other grouping of related sequences that represent your protein of interest. We provide tools for creating these prompts from multiple sequence alignments (MSAs) and for using homology search to build MSAs from a seed sequence.


Creating a prompt
-------------------

You can create a prompt either from an MSA (see below), or by directly uploading a *de novo* prompt :py:meth:`openprotein.api.poet.PoetAPI.upload_prompt`. 

To create an MSA based prompt, you can first request an MSA with :py:meth:`openprotein.api.poet.PoetAPI.create_msa` and then filter it with :py:meth:`openprotein.api.poet.MSAFuture.sample_prompt`.

Once you have a viable prompt, you can execute the PoET workflows:

* Score arbitrary sequences you provide to predict sequence fitness and rank variants. This works for substitutions and indels and allows high order variants. (for example, :py:meth:`openprotein.api.poet.PoetAPI.score`)
* Map the fitness of all single substitution variants. This is useful for designing single mutant libraries, but also for identifying mutable hotspots and designing combinatorial variant libraries. (for example, :py:meth:`openprotein.api.poet.PoetAPI.single_site`)
* Generate novel, bespoke, high order variants by sampling from the model. This is especially useful for synthetic diversification and exploring the full and potentially diverse sequence space of your protein. (for example, :py:meth:`openprotein.api.poet.PoetAPI.generate`)



.. note::
   For a practical example of using this workflow, see the :doc:`PoET workflow notebook <demos/poet_demo>`.


Index
--------------------

.. toctree::
   :maxdepth: 2

   poet
   demos/poet_demo.ipynb
