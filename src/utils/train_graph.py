import torch
from torch.nn import Sigmoid
from typing import List, Dict
from torchmetrics import Precision, Recall, Accuracy, F1Score


def train(train_data, val_data, early_stop, scheduler, num_epochs, model, criterion, optimizer, metrics_to_watch):
    best_loss = best_epoch = 0.0
    for epoch in range(num_epochs):

        # pretty printing
        print(f'{"*"*100}')
        print(f'Epoch {epoch+1:03d}/{num_epochs}')

        # check for early stopping
        if early_stop.stop:
            break

        # start training
        model.train()
        out = model(graph=train_data,
                    entries=train_data[('user', 'rates', 'artwork')].edge_label_index)
        labels = train_data[('user', 'rates', 'artwork')].edge_label.type(torch.FloatTensor)
        out = Sigmoid()(out).flatten()
        optimizer.zero_grad()
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        print(f'Train loss: {loss.item():>10.4f}')

        # start validation
        model.eval()
        with torch.no_grad():
            out = model(graph=val_data,
                        entries=val_data[('user', 'rates', 'artwork')].edge_label_index)
            labels = val_data[('user', 'rates', 'artwork')].edge_label.type(torch.FloatTensor)
            out = Sigmoid()(out).flatten()
            loss = criterion(out, labels)
            # compute metrics
            for name, metric in metrics_to_watch.items():
                print(f'{name:<20}:{metric(out, labels):>10.2f}')
            scheduler.step(loss)
            early_stop(loss, model)
            if epoch == 1 or best_loss > loss:
                best_loss = loss
                best_epoch = epoch
        print(f'{"*" * 100}')

    print(f'Best epoch: {best_epoch:04d}')
    print(f'Best loss: {best_loss:.4f}')
