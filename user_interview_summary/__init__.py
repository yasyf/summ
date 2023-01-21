# SPDX-FileCopyrightText: 2023-present Yasyf Mohamedali <yasyfm@gmail.com>
#
# SPDX-License-Identifier: MIT
import os

import pinecone

if pinecone_api_key := os.environ.get("PINECONE_API_KEY"):
    pinecone.init(api_key=pinecone_api_key)
