from torch import optim, sum, save
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


def train(model, train_ds, val_ds, train_opts, exp_dir=None):
    assert val_ds is not None
    train_dl = DataLoader(train_ds, train_opts["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, train_opts["batch_size"] * 2, shuffle=False)
    print(f"Training on {len(train_ds)} and validating on {len(val_ds)} images")

    optimizer = optim.SGD(
        model.parameters(),
        lr=train_opts["lr"],
        momentum=train_opts["momentum"],
        weight_decay=train_opts["weight_decay"]
    )

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=train_opts["step_size"],
        gamma=train_opts["gamma"]
    )

    criterion = CrossEntropyLoss()

    num_epochs = train_opts["num_epochs"]
    epoch_loss_tr = []
    epoch_acc_tr = []
    epoch_loss_val = []
    epoch_acc_val = []

    for epoch in range(num_epochs):
        model.train()
        tr_loss, train_acc = compute_epoch_stats(epoch, model, train_dl, criterion, optimizer, lr_scheduler)
        epoch_loss_tr.append(tr_loss)
        epoch_acc_tr.append(train_acc)

        model.eval()
        val_loss, val_acc = compute_epoch_stats(epoch, model, val_dl, criterion)
        epoch_loss_val.append(val_loss)
        epoch_acc_val.append(val_acc)

        print(f"[{epoch + 1}/{num_epochs}: train_loss {tr_loss:.4} val_loss {val_loss:.4} "
              f"train_acc {train_acc:.2%} val_acc {val_acc:.2%}]")

        # save model checkpoint if exp_dir is given
        if exp_dir:
            if not os.path.exists(exp_dir):
                try:
                    os.mkdir(exp_dir)
                except FileNotFoundError:
                    exit("Make sure the output directory is a valid path")
            save(model, os.path.join(exp_dir, f"checkpoint _{epoch + 1}.pt"))
    # plot the training metrics
    plot(epoch_loss_tr, epoch_acc_tr, epoch_loss_val, epoch_acc_val)
    return model


def compute_epoch_stats(epoch, model, data_loader, criterion, optimizer=None, scheduler=None):
    epoch_loss, epoch_acc = 0, 0
    for i, (mini_x, mini_y) in enumerate(data_loader):
        pred = model(mini_x).squeeze()
        loss = criterion(pred, mini_y)
        acc = sum(mini_y.eq(pred.argmax(dim=1))).item()
        epoch_loss += loss.item()
        epoch_acc += acc

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)

    epoch_loss = epoch_loss / len(data_loader)
    epoch_acc = epoch_acc / (len(data_loader) * data_loader.batch_size)
    return epoch_loss, epoch_acc


def plot(loss_tr, acc_tr, loss_val, acc_val):
    figure, (ax1, ax2) = plt.subplots(1, 2)
    n = [i + 1 for i in range(len(loss_tr))]
    acc_tr = [x * 100 for x in acc_tr]
    acc_val = [x * 100 for x in acc_val]

    ax1.plot(n, loss_tr, 'bs-', markersize=6, label="train")
    ax1.plot(n, loss_val, 'rs-', markersize=6, label="val")
    ax1.legend(loc="upper right")
    ax1.set_title("Losses")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")

    ax2.plot(n, acc_tr, 'bo-',  markersize=6, label="train")
    ax2.plot(n, acc_val, 'ro-', markersize=6, label="val")
    ax2.legend(loc="upper right")
    ax2.set_title("Accuracy")
    ax2.set_ylabel("Acc (%)")
    ax2.set_xlabel("Epoch")
    plt.savefig("net-trained.png")
    plt.show()
