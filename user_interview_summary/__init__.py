# SPDX-FileCopyrightText: 2023-present Yasyf Mohamedali <yasyfm@gmail.com>
#
# SPDX-License-Identifier: MIT
import os

import pinecone, redis_om, logging
from redis_om import checks, Migrator

if pinecone_api_key := os.environ.get("PINECONE_API_KEY"):
    pinecone.init(api_key=pinecone_api_key)

try:
    if not checks.has_redisearch():
        raise TypeError
except TypeError as e:
    raise Exception("Redisearch not installed. Try `brew install redis-stack/redis-stack/redis-stack") from e
else:
    Migrator().run()
