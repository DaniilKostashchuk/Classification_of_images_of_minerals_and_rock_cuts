class_counts = torch.bincount(torch.tensor([label for _, label in full_dataset]))
class_weights = 1.0 / class_counts.float()  
class_weights = class_weights.to(device)
