import os
import time
import sys

import torch
import torchvision

import dataset
import model

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True


class Logger(object):
    def __init__(self, filename="log_CUB_my_idea_CAM_PAM_2.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    data_path = './CUB_200_2011'

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    options = {
        'base_lr': 1e-2,
        'batch_size': 32,
        'epochs': 100,
        'weight_decay': 1e-4,
    }

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size=448,
                                                 scale=(0.8, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225)),

    ])
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(448, 448)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225)),
    ])
    train_data = dataset.CUB200(
        root=data_path, train=True,
        transform=train_transforms, download=True)
    test_data = dataset.CUB200(
        root=data_path, train=False,
        transform=test_transforms, download=True)

    net = model.BCNN(num_class=200).cuda()

    print(net)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(
        net.parameters(), lr=options['base_lr'],
        momentum=0.9, weight_decay=options['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                           patience=8, verbose=True, threshold=1e-4)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=options['batch_size'], shuffle=True, num_workers=4, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=options['batch_size'], shuffle=False, num_workers=2, pin_memory=False)

    """Train the network."""
    print('Training.')
    net.train()
    net.cuda()
    best_acc = 0.0
    best_epoch = None
    print('Epoch\tTrain loss\tTrain acc\tTest acc\tTime\n')
    for t in range(options['epochs']):
        epoch_loss = []
        num_correct = 0
        num_total = 0
        num_count = 0
        tic = time.time()
        for instances, labels in train_loader:
            num_count += 1
            instances = instances.cuda()
            labels = labels.cuda()

            score = net(instances)
            loss = criterion(score, labels)
            with torch.no_grad():
                epoch_loss.append(loss.item())

                prediction = torch.argmax(score, dim=1)
                num_total += labels.size(0)
                num_correct += torch.sum(prediction == labels).item()
            if num_count % 30 == 0:
                print('\n')
                print(loss.item(), 100 * num_correct / num_total)

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('backward\n')
            del instances, labels, score, loss, prediction
        train_acc = 100 * num_correct / num_total

        """test the data"""
        with torch.no_grad():
            net.eval()
            num_correct = 0
            num_total = 0
            for instances, labels in test_loader:

                instances = instances.cuda()
                labels = labels.cuda()

                score = net(instances)

                prediction = torch.argmax(score, dim=1)
                num_total += labels.size(0)
                num_correct += torch.sum(prediction == labels).item()
            test_acc = 100 * num_correct / num_total

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = t + 1
            print('*', end='')
            # save_path =
            # torch.save(self._net.state_dict(), save_path)
        toc = time.time()
        print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f min' %
              (t + 1, sum(epoch_loss) / len(epoch_loss), train_acc,
               test_acc, (toc - tic) / 60))
        print('Best at epoch %d, test accuaray %4.2f' % (best_epoch, best_acc))
        print("--------------------------------------------")
        scheduler.step(test_acc)
    print('Best at epoch %d, test accuaray %4.2f' % (best_epoch, best_acc))


if __name__ == '__main__':
    sys.stdout = Logger()
    main()
