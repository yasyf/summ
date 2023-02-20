# Summ

[![PyPI - Version](https://img.shields.io/pypi/v/summ.svg)](https://pypi.org/project/summ)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/summ.svg)](https://pypi.org/project/summ)

-----

Summ uses ChatGPT to provide intelligent question-answering and search capabilities across user transcripts!

Easily surface insights and summarize facts across various dimensions such as department, industry, and role. With the help of natural language processing, the tool can understand and respond to complex questions and queries, making it easy for users to find the information they need.

A tool by [@markiewagner](https://github.com/markiewagner) and [@yasyf](https://github.com/yasyf).

[![asciicast](https://asciinema.org/a/6dNMwGgNrmBrnFjyFjbJJ2xLR.svg)](https://asciinema.org/a/6dNMwGgNrmBrnFjyFjbJJ2xLR)

## Requirements

You'll need an instance of [Redis Stack](https://redis.io/docs/stack/get-started/install/) running.

```console
$ brew install --cask redis-stack/redis-stack/redis-stack-server
$ brew install yasyf/summ/redis-stack
$ brew services start yasyf/summ/redis-stack
```

You'll also need to set three environment variables: `OPENAI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_ENVIRONMENT`.


## Installation

The easiest installation uses `pip`:

```console
$ pip install summ
```

If you prefer to use `brew`:

```console
$ brew install yasyf/summ/summ
```

**n.b `summ` requires Python 3.10+.**

## Demo

You can confirm that `summ` installed properly by running the built-in example.

```console
$ summ-example
```

## Quickstart

First, create a new project with:

```
$ summ init /path/to/project
$ cd /path/to/project
```

An example implementation can now be found at `/path/to/project/__init__.py`.

As you can see, you don't need to do any configuration to start using `summ`. We simply use `summ.Pipeline.default` and pass a path to a directory with text files.

However, the tool works much better when users are tagged. In order to do so, you need to specify two things:

1. The categories of tags (and the tags within each category).
2. A prompt directing how to apply the tags of a given category.

You can see an example of this at [`summ/examples/otter`](summ/examples/otter).

## Docs

Check out the [`summ/examples`](summ/examples) directory for some samples, or dive into the full docs at [summ.readthedocs.io](https://summ.readthedocs.io/en/latest/).

## License

`summ` is distributed under the terms of the [AGPL 3.0](https://spdx.org/licenses/AGPL-3.0-only.html) license.
