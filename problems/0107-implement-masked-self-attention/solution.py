import numpy as np
from scipy.special import softmax

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	temp1 = Q @ K.T
	scale_num = np.sqrt(K.shape[-1])
	temp2 = temp1/scale_num
	# mask = np.triu(
	# 	np.full((temp2.shape[0],temp2.shape[1]), -np.inf),
	# 	k=1
	# )
	temp2 = temp2 + mask
	temp3 = softmax(temp2, axis=-1)
	temp4 = temp3 @ V
	return temp4
