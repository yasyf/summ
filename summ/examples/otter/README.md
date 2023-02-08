# Otter.ai User Interview Example

This directory has an example of using `summ` with text files exported from [otter.ai](https://otter.ai).

The conversations included discuss [Cronutt](https://magazine.ucsf.edu/cellular-transplant-therapy-pioneered-ucsf-reverses-sea-lions-epilepsy).

## Setup

### Classes

The class `MyClasses` in [`implementation/classes.py`](implementation/classes.py) sets out four categories of tags: job title, company type, department, and industry.

### Classifiers

The classifiers in [`implementation/classifier.py`](implementation/classifier.py) use simple parameters to define a prompt for each category of tags. It is normally sufficient to simply provide `CATEGORY`, `VARS`, and `EXAMPLES`. You may also optionally specify a `PREFIX` or `SUFFIX` for the prompt.

### CLI

Finally, in [`implementation/__init__.py`](implementation/__init__.py), we:

1. Ensure our classifiers are imported
2. Construct a `Summ` object, passing a `Path` to our training data.
3. Construct a custom `Pipeline` object which specifies the otter.ai import format.
4. Pass these two to `summ.cli.CLI`, which creates a command line interface for us.

## Usage

### TUI

To run the Terminal UI, simply do:

```bash
$ python -m implementation
```

You can also run the steps non-interactively, as shown below.

### Populate

Now, to populate our model, we can do:

```bash
$ python -m implementation populate
```

### Query

And to query it:

```bash
$ python -m implementation query "What kind of animal is Cronutt?"
Cronutt is a California sea lion, a species of marine mammal.
```
