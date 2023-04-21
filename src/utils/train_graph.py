import torch
from torch.nn import Sigmoid


def train(train_data, val_data, early_stop, scheduler, num_epochs, model, criterion, optimizer, metrics_to_watch,
          verbose=False, device = 'cuda'):
    best_loss = best_epoch = 0.0
    for epoch in range(num_epochs):

        if verbose:
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
        labels = train_data[('user', 'rates', 'artwork')].edge_label.type(torch.FloatTensor).to(device)
        out = Sigmoid()(out).flatten().to(device)
        optimizer.zero_grad()
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        if verbose:
            print(f'Train loss: {loss.item():>10.4f}')

        # start validation
        model.eval()
        with torch.no_grad():
            out = model(graph=val_data,
                        entries=val_data[('user', 'rates', 'artwork')].edge_label_index)
            labels = val_data[('user', 'rates', 'artwork')].edge_label.type(torch.FloatTensor).to(device)
            out = Sigmoid()(out).flatten().to(device)
            loss = criterion(out, labels)

            if verbose:
                # compute metrics
                for name, metric in metrics_to_watch.items():
                    print(f'{name:<20}:{metric(out, labels):>10.2f}')
            scheduler.step(loss)
            early_stop(loss, model)
            if epoch == 1 or best_loss > loss.item():
                best_loss = loss.item()
                best_epoch = epoch

        if verbose:
            print(f'{"*" * 100}')

    if verbose:
        print(f'Best epoch: {best_epoch:04d}')
        print(f'Best loss: {best_loss:.4f}')
    return best_loss