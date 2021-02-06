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