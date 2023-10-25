#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 19:10:45 2023

@author: sarpaltunel
"""

from cat import Cat

cat1 = Cat("Kittosaurus Rex")
cat2 = Cat("Snowball IX")

cat1.greet(cat2)
cat2.greet(cat1)