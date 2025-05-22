from torch import optim, sum
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def train(model, train_ds, train_opts):
    """
    fits an instance of a sequential model on some data

    Arguments
    --------
    model (nn.Sequential): the model to fit
    train_ds (TensorDataset): the train input features and corresponding labels
    train_opts (Dictionary): specifications of the training schedule. Must have the following keys
                            1. num_epochs: value is the number of training epochs
                            2. lr: value is the initial learning rate
                            3. momentum: value is the optimizer's momentum
                            4. weight_decay: value is the weight decay factor
                            5: batch_size: value is the mini-batch size
                            6: step_size: The learning rate step size
                            7. gamma: the learning rate decay factor
    Returns
    -------
    model (nn.Sequential): The trained model
    """

    print(f"Training on {len(train_ds)} examples")
    optimizer = optim.SGD(model.parameters(),
                          lr=train_opts['lr'],
                          momentum=train_opts['momentum'],
                          weight_decay=train_opts['weight_decay'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, train_opts['step_size'], gamma=train_opts['gamma'])
    criterion = CrossEntropyLoss()

    train_dl = DataLoader(train_ds, batch_size=train_opts['batch_size'], shuffle=True, drop_last=False)
    num_epochs = train_opts['num_epochs']
    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        tr_loss = tr_acc = 0
        for x_tr, y_tr in train_dl:
            model.train()
            pred_y = model(x_tr)
            loss = criterion(pred_y, y_tr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch=epoch)
            acc = sum(y_tr.eq(pred_y.argmax(dim=1))).item()
            tr_acc += acc
            tr_loss += loss.item()
        tr_loss = tr_loss/len(train_dl)
        tr_acc = tr_acc/len(train_ds)
        print(f"[{epoch + 1}/{num_epochs}: loss {tr_loss:.5}  accuracy {tr_acc:.2%}]")

        losses.append(tr_loss)
        accuracies.append(tr_acc)
    plot(loss_tr=losses, acc_tr=accuracies)
    return model


def plot(loss_tr, acc_tr):
    acc_tr = [x * 100 for x in acc_tr]
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    n = [i + 1 for i in range(len(loss_tr))]
    ax1.plot(n, loss_tr, 'rs-', markersize=6)
    ax1.set_title("Losses")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.plot(n, acc_tr, 'bo-', markersize=6)
    ax2.set_title("Accuracy")
    ax2.set_ylabel("Acc (%)")
    ax2.set_xlabel("Epoch")
    plt.savefig("train_metrics.png")
    # plt.show()
