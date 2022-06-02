import torchvision
import cv2


class ImageNetCustomDataset(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label