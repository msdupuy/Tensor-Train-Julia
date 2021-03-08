---
use.Math: true
---

# Tensor Trains

This package aims to implement basic routines in Julia for the tensor train (TT) format.
It is still in the early stage of development. At the moment, 
the following features are available:
- compute tensor train representations of vectors and matrices
- basic TT rounding algorithms
- solve linear systems or smallest eigenvalue problems for problems in the tensor train format using the Alternating Linear Scheme (ALS or one-site DMRG) or the Modified Alternating Linear Scheme (MALS or two-site DMRG). The implementation is based on 
    - Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider. "The alternating linear scheme for tensor optimization in the tensor train format." SIAM Journal on Scientific Computing 34.2 (2012): A683-A713.
- generate the TT representation of Hamiltonians of the form 
<div class="math">
$$ H = \sum_{i,j} h_{ij} (a_i^\dagger a_j + c.c.) +  \sum_{i,j,k,l} V_{ijkl} (a_i^\dagger a_j^\dagger a_k a_l + c.c.) $$
</div>
- compute ground-states of the 1D or 2D Hubbard model (with open or periodic boundary conditions)
