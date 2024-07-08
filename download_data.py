#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 16:07:37 2024

@author: guoxing.lan
"""

from datasets import load_dataset

# docs = load_dataset('irds/nyt', 'docs')
load_dataset(
            "imdb",
            cache_dir="./.cache"
        )
