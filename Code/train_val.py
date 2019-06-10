import torch.nn as nn
from data_loader import *
import argparse
from tensorboardX import SummaryWriter


def train(model, optimizer, criterion, train_loader, val_loader, writer, epoch):
    global experiment
    model.train()

    idx = 0
    loss_total = 0  # average loss over 100 iterations

    for e in range(epoch):
        for bID, data in enumerate(train_loader):
            imgs = data['img'].to(var.device)
            ques = data['ques'].to(var.device)
            ans = data['ans'].to(var.device)

            pred = model(imgs, ques)
            loss = criterion(pred, ans)

            print("debug loss:", loss.item())

            loss_total += loss.detach().clone().cpu().data.numpy()
            idx += 1

            if idx % 100 == 0:
                writer.add_scalar('train/loss', loss_total / 100, idx)
                print("Iteration {}: loss is {}".format(idx, round(loss_total / 100, 2)))
                loss_total = 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, val_acc = val(model, criterion, val_loader)
        model.train()

        print("Epoch {}: loss is {}, acc is {}".format(e, round(val_loss, 2), round(val_acc, 2)))
        writer.add_scalar('val/loss', val_loss, e)
        writer.add_scalar('val/acc', val_acc, e)

        # save model
        dir = os.path.join(var.model_dir, 'exp' + str(experiment), str(e))

        if not os.path.exists(dir):
            os.makedirs(dir)

        path = os.path.join(dir, var.model_name)
        save_ckpt(model, optimizer, val_loss, val_acc, path)


def val(model, criterion, loader):
    with torch.no_grad():
        cnt = 0
        total_len = 0
        acc = 0.0
        total_val_loss = 0.0

        for bID, data in enumerate(loader):
            imgs = data['img'].to(var.device)
            ques = data['ques'].to(var.device)
            ans = data['ans'].to(var.device)

            pred = model(imgs, ques)
            val_loss = criterion(pred, ans)

            pred = pred.detach().clone().cpu().data.numpy()
            total_val_loss += val_loss.detach().clone().cpu().data.numpy()
            ans = ans.clone().cpu().data.numpy()

            # use a different metric to calculate acc here (count whether the prediction meets the best answer)
            pred_idx = np.argmax(pred, 1)
            acc += np.sum(pred_idx == ans)

            cnt += 1
            total_len += len(ans)

        acc = float(acc) / total_len
        total_val_loss /= cnt

    return total_val_loss, acc * 100


class experiment_runner():
    def __init__(self, exp_id):
        self.exp_id = exp_id

    def run(self):
        if self.exp_id == 1:
            from experiment1 import run_experiment
            run_experiment()
        elif self.exp_id == 2:
            from experiment2 import run_experiment
        else:
            raise NotImplemented

        run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=int, default=1)

    args = parser.parse_args()

    experiment = args.e

    runner = experiment_runner(experiment)
    runner.run()
