# SPDX-FileCopyrightText: 2023-present Yasyf Mohamedali <yasyfm@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-only
import os

import click

click.secho("Starting up, please hold...", fg="yellow")

import langchain

# pinecone makes network requests when imported
import pinecone
from langchain.cache import RedisCache
from redis import Redis
from redis_om import Migrator, checks

from summ.pipeline import Pipeline as Pipeline
from summ.summ import Summ as Summ

if pinecone_api_key := os.environ.get("PINECONE_API_KEY"):
    pinecone.init(
        api_key=pinecone_api_key,
        environment=os.environ.get("PINECONE_ENVIRONMENT", "us-west1-gcp"),
    )

langchain.llm_cache = RedisCache(redis_=Redis(db=1))

try:
    if not checks.has_redisearch():
        raise TypeError
except TypeError as e:
    raise Exception(
        "Redisearch not installed. Try `brew reinstall yasyf/summ/redis-stack"
    ) from e
else:
    Migrator().run()
