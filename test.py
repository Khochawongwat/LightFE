from loss import VGGPerceptualLoss, PSNR
import itertools
import tqdm

def test(model, loader, device = "cuda"):
    model.eval()
    loss = VGGPerceptualLoss().to(device)
    model = model.to(device)
    print(f"Training is {model.training}")
    if model.training:
        return "Model is still training."
    total_loss, total_psnr = 0, 0
    total_n = 0
    for i, batch in enumerate(loader):
        x = y = batch
        print(len(x))
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss_value = loss(output, y)
        total_loss += loss_value.item()
        total_psnr += PSNR(output, y)
        total_n += 1

    return total_psnr/total_n, total_loss/total_n