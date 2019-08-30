# Check if Numba is available
try:
	import numba
	NUMBA_AVAILABLE = True
except ImportError:
	NUMBA_AVAILABLE = False