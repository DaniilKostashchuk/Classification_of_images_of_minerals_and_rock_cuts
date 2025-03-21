dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
}

dataset_sizes = {'train': train_size, 'val': val_size}
class_names = full_dataset.classes  # Названия классов
num_classes = len(class_names)
