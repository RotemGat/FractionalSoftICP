import torch

from fractional_soft_icp import fractional_soft_icp_loss


torch.manual_seed(0)
target = torch.randn(256, 3)
source = target + torch.tensor([0.7, -0.4, 0.2])

translation = torch.nn.Parameter(torch.zeros(3))
optimizer = torch.optim.Adam([translation], lr=0.05)

with torch.no_grad():
    initial_loss = fractional_soft_icp_loss(
        source,
        target,
        fraction=0.75,
        sigma=0.1,
        chunk_size=128,
    )

for step in range(200):
    optimizer.zero_grad()
    aligned_source = source + translation
    loss = fractional_soft_icp_loss(
        aligned_source,
        target,
        fraction=0.75,
        sigma=0.1,
        chunk_size=128,
    )
    loss.backward()
    optimizer.step()

with torch.no_grad():
    final_loss = fractional_soft_icp_loss(
        source + translation,
        target,
        fraction=0.75,
        sigma=0.1,
        chunk_size=128,
    )

print("Learned translation:", translation.detach().tolist())
print(f"Loss: {initial_loss.item():.6f} -> {final_loss.item():.6f}")
