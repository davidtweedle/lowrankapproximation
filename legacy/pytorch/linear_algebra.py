import torch

def inverse_pth_root(A, L, b, p, m, low_rank):
  """
  A: A ** -p = L
  p: positive integer
  vec: vector
  returns
  A' such that
  A' ** -p = M + bb^T
  """
  U, alphas, betas, u = lanczos(A, b, m)
  # compute Krylov subspace corresponding to A,b
  # AU = UG + beta ue*
  M = U.T @ L
  M = M @ U
  c = U.T @ b
  X = -pth_root_from_alpha_beta(alphas, betas)
  alphas[0] += torch.norm(b)
  X += pth_root_from_alpha_beta(alphas_plus_b, betas)
  return A + (U @ X) @ U.T

def lanczos(A, b, m):
  n = torch.size()[0] 
  u_prev = torch.zeros(n)
  u = torch.div(b, torch.norm(b))
  U = u[:, None]
  betas = torch.zeros(m)
  alphas = torch.zeros(m)
  for j in range(m):
    w = A @ u - betas[j] * u_prev
    alphas[j] = u.T @ w
    w -= alphas[j] * u
    if j < m:
      betas[j + 1] = torch.norm(w)
      u_prev = u
      u = torch.div(w, betas[j+1])
      U = torch.cat((U, u[:,None]), -1)
  return U, alphas, betas


def pth_root_from_alpha_beta(alphas, betas, p):
  """
  Computes the inverse pth root of a tridiagonal matrix
  with diagonal corresponding to alphas
  and above and below the diagonal are betas
  """
  A = torch.diag(alphas, diagonal=0)
  A += torch.diag(betas[1:], diagonal=1)
  A += torch.diag(betas[1:], diagonal=-1)
  L, Q = torch.linalg.eigh(A)
  inv_pth_root_diag = torch.pow(L, -1/p)
  return (Q @ torch.diag(inv_pth_root_diag)) @ Q.T
