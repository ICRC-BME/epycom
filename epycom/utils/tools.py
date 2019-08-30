# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports

# Third pary imports

# Local imports

def conditional_jitdecorate(condition, jit_kwargs):
	if condition:
		try:
			from numba import jit
			return jit(**jit_kwargs)
		except ImportError:
			return lambda x: x
	else:
		return lambda x: x
