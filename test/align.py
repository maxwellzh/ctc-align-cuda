
from ctc_align._C import align_
import torch
import time


if __name__ == "__main__":
    torch.manual_seed(0)
    n = 1
    device = 9
    # torch.cuda.set_device(device)


    for i in range(n):
        # N, T, V, K = 2, 12, 8, 4
        # lx = torch.randint(max(2, T//4), T, (N, ),
        #                    device=device, dtype=torch.int32)
        # lx += (T-lx.max())
        # p = torch.randn(N, T, V, device=device)
        # p[:, :, 0] += 3
        # p = p.softmax(dim=-1)
        # samples = torch.multinomial(
        #     p.view(-1, p.size(-1)), K, replacement=True)
        # samples = samples.view(N, T, -1).transpose_(1,
        #                                             2).contiguous().view(-1, T)
        # l_samples = lx.unsqueeze(1).repeat(1, K).contiguous().view(-1)

        samples = torch.tensor(
            [
                [1, 0, 1, 2, 2, 6, 0, 1, 1, 2, 3, 3, 3, 5],
                [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 5],
                [1, 0, 0, 0, 2, 7, 0, 1, 0, 2, 3, 3, 3, 5]
            ],
            dtype=torch.int64, device=device
        )
        l_samples = torch.tensor([12, 14, 14], dtype=torch.int, device=device)
        print("python:", samples.device)
        
        print(torch.cuda.max_memory_allocated(device))
        print(samples.shape)
        print(samples[:5])
        print(l_samples)
        samples, lens = align_(samples, l_samples)
        print(samples[:5], lens[:5])
        pass
