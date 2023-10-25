#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 19:10:36 2023

@author: sarpaltunel
"""

class Cat:
    def __init__(self, name):
        self.name = name
        
    def greet(self, other_cat):
        print(f"Hellooo, I am {self.name}! I see you are also a cool fluffy kitty {other_cat.name}, let’s together purr at the human, so that they shall give us food.")