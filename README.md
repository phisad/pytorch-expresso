# pytorch-expresso

A high-level framework for PyTorch to quickly set up a training pipeline by using configurable experiments.

**Read to use.** The framework is supposed set plausible defaults to run experiments quickly.

**Ready to adjust.** The framework always allows for adjustment and extension where necessary.

**Everthiny is a callback.** This framework makes heavily use of the "callback idea", so that almost everything is
implemented as an attachable callback. The users of this framework are also supposed to customize in this way.

**Only one truth.** Either modules like callbacks are fully configured or fully implemented. The implementation is used,
when not configuration is used.

# The main flow

    +--------+      +--------------+
    | Steps  | <->  |  Callbacks   |
    +--------+      +--------------+

# Experiment configuration

This project uses a highly customizable configuration framework.

The configuration structure is straight-forward:

    configs
        +-- cometml
        +-- datasets
        +-- models
        +-- tasks
        +-- experiments

**cometml**: Define a comet-ml user to run experiments. Follow the sample.json. Then you can run the experiments using
the -u myuser option. All added user files are git-ignored. Simply copy and rename the sample.json to myuser.json and
provide the api-key, workspace and project name.

**dataset**: Defines the dataset provider to use and the split structure and the vocabulary file. The dataset provider
is dynamically loaded during the experiment using the configured package and class name.

**models**: Define available models and their hyper-parameters for the experiments. The model is dynamically loaded
during the experiment using the configured package and class name. Models link the network architecture to according
hyper-parameters. Thus an architecture with different hyper-parameters requires an additional file. This automatically
documents the training procedure (but might be problematic for automatic hyper-parameter search).

**tasks**: Define task specific parameters that are necessary for both models and dataset providers. The task parameters
are injected in the configured dataset and model e.g. vocabulary size. Tasks allow models and providers to behave
slightly different while sharing most of the code along experiments. For example, a model architecture might stay the
same while the classifcation task changes (source block vs directions).

**experiments**: Combines *dataset, models and tasks* and allows to introduce further *params* to the training
procedure (gpu, epochs, batch size). The experiment params might also include certain references to specific classes
for ["step_fn","loss_fn","optim_fn"] all coming with potential *kwargs* to be injected.

The previous mentioned can all be directly included in the experiment config for better readability. Still, when certain
modules are repeatitly used, then they can also be put into an own file. For example a certain task that is the same for
all experiments. The modules are then referenced by their file names.

These file references are dynamically loaded during the experiment assuming the config top dir is given in the
ConfigurationLoader. The values for ["model", "dataset", "task", "env", "callbacks", "savers"] which end with json are
interpreted as seperate files and loaded as dict-like or list objects.

Experiment configurations allow for example to run:

- different models on the same task and dataset (fnn vs rnn)
- different tasks using the same model on the same datasets (source vs target loation)
- different datasets using the same model and (possibly different) task (semantics vs locations)

# Install

Note: The framework already depends on torch, torchvision and comet-ml, so that the only requirement would be this
project.

Checkout the sources

`$> git clone git@github.com:pytorch-expresso.git`

Go to the python environment (preferebly anaconda) you want pytorch-expresso

`$> conda activate my-conda-env`

Go to the sources directory

`$> cd pytorch-expresso`

Execute the setup script from within the python environment

`$> python setup.py install`

The framework is now available to all projects which share the python environment.

Happy Coding!