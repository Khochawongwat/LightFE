import os 
import torch
import tqdm
import torchvision.transforms as transforms
import random
from utils import DecreaseResolution, RandomTransform
import datetime

def train(model, dataloader, loss, optimizer, size,  epochs = 3, device="cuda", iter = 100):
    
    loss = loss.to(device)
    model = model.to(device)
    model.train()

    print(f"Training is {model.training}")

    os.makedirs("models", exist_ok=True)

    for epoch in range(0, epochs):

        now = datetime.datetime.now()
        time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

        print(f"Epoch:{epoch + 1}")

        with open(f'./logs/{time_string}({size})({epoch}).txt', 'w') as log:

            losses = []

            depths = []

            for i, batch in tqdm.tqdm(enumerate(dataloader)):

                optimizer.zero_grad()

                x = y = batch.clone()
                
                depth = random.randint(1, 3)

                depths.append(depth)

                x_transform = transforms.Compose(
                    [
                        DecreaseResolution(depth = depth),
                    ]
                )

                y_transform = RandomTransform(
                    [
                        transforms.RandomHorizontalFlip(p=random.uniform(0.0, 1.0)),
                        transforms.RandomVerticalFlip(p=random.uniform(0.0, 1.0)),
                        transforms.RandomRotation(random.randint(0, 360)),
                    ]
                )

                y = y_transform(y)
                x = x_transform(y)

                x, y = x.to(device), y.to(device)

                output = model(x)
                loss_value = loss(output, y)
                loss_value.backward()
                optimizer.step()

                losses.append(loss_value.item())

                if len(losses) > iter:
                    losses.pop(0)
                    
                if i % iter == 0:
                    avg_loss = sum(losses) / len(losses)
                    log.write(f"Size: {size}, Epoch: {epoch + 1}, Iteration: {i}, Average Loss: {avg_loss}\n")

                if i % iter*100 == 0 and i != 0:
                    log.write(f"Saving model {epoch + 1}_{i}\n")
                    torch.save(model.state_dict(), f"./models/model({size})_{epoch}.pth")

                torch.cuda.empty_cache()
                del x, y, output, loss_value

            log.write(str(depths))

    model = model.to("cpu")
    return model
