for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Сохраняем предсказания и метки для расчета метрик
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        # Расчет Precision, Recall и F1 Score
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Сохранение метрик
        if phase == 'train':
            Epoch_loss_train_list.append(epoch_loss)
            Epoch_acc_train_list.append(epoch_acc.item())
            Precision_train_list.append(precision)
            Recall_train_list.append(recall)
            F1_train_list.append(f1)
        else:
            Epoch_loss_val_list.append(epoch_loss)
            Epoch_acc_val_list.append(epoch_acc.item())
            Precision_val_list.append(precision)
            Recall_val_list.append(recall)
            F1_val_list.append(f1)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} '
              f'Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}')

        # Сохранение лучшей модели
        if phase == 'val':
            if epoch_acc > best_val_accuracy:
                best_val_accuracy = epoch_acc
                no_improve_epochs = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with accuracy: {best_val_accuracy:.4f}")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= early_stop_patience:
                    print("Early stopping triggered")
                    break

    # Обновление learning rate
    scheduler.step()

    # Сохранение последней модели
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model saved at epoch {epoch + 1}")

    # Остановка обучения, если learning rate слишком мал
    if optimizer.param_groups[0]['lr'] < 1e-10:
        print("Learning rate is too small. Stopping training.")
        break

print("Training complete.")
