# Summ

[![PyPI - Version](https://img.shields.io/pypi/v/summ.svg)](https://pypi.org/project/summ)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/summ.svg)](https://pypi.org/project/summ)

-----

Summ uses ChatGPT to provide intelligent question-answering and search capabilities across user transcripts!

Easily surface insights and summarize facts across various dimensions such as department, industry, and role. With the help of natural language processing, the tool can understand and respond to complex questions and queries, making it easy for users to find the information they need.

A tool by @markiewagner and @yasyf.

## Installation

```console
pip install summ
```

You'll need an instance of [Redis Stack](https://redis.io/docs/stack/get-started/install/) running. We've found that `brew install redis-stack/redis-stack/redis-stack` is the fastest way to get up and running.

You'll also need to set two environment variables: `OPENAI_API_KEY`, and `PINECONE_API_KEY`

## Docs

**TODO: Docs on how to customize**

## License

`summ` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
