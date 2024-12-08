import torch
import torchvision.transforms.v2 as v2
import torchvision.transforms as transforms
def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            inputs, targets, *_ = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy




def add_normal_noise(input_tensor, mean: int = 0, std: float = 0.5):
    # Create the tensor containing the noise
    noise_tensor = torch.empty_like(input_tensor)
    noise_tensor.normal_(mean=mean, std=std)
    # Add noise to input tensor and return results
    return input_tensor + noise_tensor


def wrap_add_normal_noise(std=0.3):
    def add_normal_noise(tensor):
        return tensor + torch.randn(tensor.size()) * std
    return add_normal_noise

class DataTransforms:
    def __init__(self):
        self.train_transform_1 = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomCrop(size=100, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.train_transform_2 = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.train_transform_3 = transforms.Compose([
            transforms.RandomRotation(degrees=45),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.RandomResizedCrop(size=100, scale=(0.7, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.train_transform_4 = transforms.Compose([
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            v2.Lambda(lambd=wrap_add_normal_noise(std=0.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.preprocess_transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
