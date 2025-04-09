import matplotlib.pyplot as plt


def plot_losses(history) -> None:
    train_losses = [x["train_loss"] for x in history]
    val_losses = [x["val_loss"] for x in history]
    train_acc = [x["train_acc"] for x in history]
    val_acc = [x["val_acc"] for x in history]
    plt.plot(train_losses, "-bx")
    plt.plot(val_losses, "-rx")
    plt.plot(val_acc, "-gx")
    plt.plot(train_acc, "-mx")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid()
    plt.legend(["train_loss", "val_loss", "val_acc", "train_acc"])
    plt.title("Loss vs. NO. of epochs")
