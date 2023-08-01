# Welcome to the docs!

Welcome to OpenProtein.AI! 

OpenProtein.AI is a powerful platform that seamlessly integrates state-of-the-art machine learning and generative models into protein engineering workflows. 

We help you to design better proteins faster by giving you access to cutting-edge protein language models, prediction and design algorithms, as well as data management and model training tools. Easily build and deploy high-performance Bayesian protein function predictors, or apply generative protein language models to design sequence libraries, all via our integrated platform. OpenProtein.ai can be accessed via both web App and API, making it great for both biologists and protein engineers, as well as data scientists and software engineers. 

## Job system

Here’s a quick overview of our jobs system. 

The platform operates as an asynchronous POST and GET framework. By initiating a task with our python client, the system schedules the job and promptly returns a response (including a unique Job ID). This approach ensures that tasks, even those with longer processing times, do not require immediate waiting. 

When you submit a task, such as `session.poet.create_msa()` you will have a Future Class returned for results tracking and access. You can check a job's status by calling the `refresh()` and `done()` methods on this Future Class. Additionally, you can call `wait()` to wait and return the results (or `get()` if results are already completed). 

Additionally, you can resume a workflow by `load_job` and using the unique job ID you got at task execution. `load_job` will resume your workflow by returning a Future Class for further manipulation (as above). 

Read on for more specific documentation, and check our demo workflows to see more!

