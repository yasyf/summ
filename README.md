# Summ

[![PyPI - Version](https://img.shields.io/pypi/v/summ.svg)](https://pypi.org/project/summ)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/summ.svg)](https://pypi.org/project/summ)

-----

Summ uses ChatGPT to provide intelligent question-answering and search capabilities across user transcripts!

Easily surface insights and summarize facts across various dimensions such as department, industry, and role. With the help of natural language processing, the tool can understand and respond to complex questions and queries, making it easy for users to find the information they need.

A tool by [@markiewagner](https://github.com/markiewagner) and [@yasyf](https://github.com/yasyf).

[![asciicast](https://asciinema.org/a/V2G8wyEfucFcU2bSr6eOCWOfP.svg)](https://asciinema.org/a/V2G8wyEfucFcU2bSr6eOCWOfP)

## Requirements

You'll need an instance of [Redis Stack](https://redis.io/docs/stack/get-started/install/) running. If you install `summ` using `brew`, this will be taken care of for you.

If you install `summ` using `pip`, this is the easiest way to get Redis up and running:

```console
$ brew install yasyf/summ/redis-stack
$ brew services start yasyf/summ/redis-stack
```

You'll also need to set three environment variables: `OPENAI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_ENVIRONMENT`.


## Installation

The easiest installation uses `brew`:

```console
$ brew install yasyf/summ/summ
```

If you prefer to use `pip`:

```console
$ pip install summ
```

## Quickstart

You don't need to do any configuration to start using `summ`. Simply use `summ.Pipeline.default` and pass a path to a directory with text files.

However, the tool works much better when users are tagged. In order to do so, you need to specify two things:

1. The categories of tags (and the tags within each category).
2. A prompt directing how to apply the tags of a given category.

You can see an example of this at [`examples/otter`](examples/otter).

## Docs

Check out the [`examples`](examples) directory for some samples, or dive into the full docs at [summ.readthedocs.io](https://summ.readthedocs.io/en/latest/).

## License

`summ` is distributed under the terms of the [AGPL 3.0](https://spdx.org/licenses/AGPL-3.0-only.html) license.
