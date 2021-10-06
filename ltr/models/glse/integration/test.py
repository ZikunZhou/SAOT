import torch
def gen_adj(As):
    """
    args:
        As (Tensor) - shape = [batch, N, N]
    """
    A_hats = As + torch.eye(As.shape[1]).unsqueeze(0)
    print(A_hats)
    D_hats = torch.pow(A_hats.sum(2), -0.5)
    print(D_hats)
    D_hats = torch.stack([torch.diag(D_hat) for D_hat in D_hats], dim=0)
    print(D_hats)
    normed_adjmat = torch.matmul(torch.matmul(A_hats, D_hats).transpose(dim0=1, dim1=2), D_hats)
    print(torch.matmul(A_hats, D_hats))
    print(torch.matmul(A_hats, D_hats).transpose(dim0=1, dim1=2))
    return normed_adjmat



if __name__ == "__main__":
    A = torch.tensor([[0.,0.,1.],[1.,0.,1.], [1.,0.,0.]])

    print(A)
    gen_adj(A)
