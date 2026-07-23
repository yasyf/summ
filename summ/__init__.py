# SPDX-FileCopyrightText: 2023-present Yasyf Mohamedali <yasyfm@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-only
import click

click.secho("Starting up, please hold...", fg="yellow")

import langchain

from langchain.cache import RedisCache
from redis import Redis

from summ.pipeline import Pipeline as Pipeline
from summ.summ import Summ as Summ

langchain.llm_cache = RedisCache(redis_=Redis(db=1))
