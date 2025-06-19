import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from opensoundscape.ml import cnn_architectures


def identity(x):
    return x


def figsize(w, h):
    plt.rcParams["figure.figsize"] = [w, h]


# Define the projection head (e.g., 2-layer MLP with ReLU activation)
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # BatchNorm after hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),  # BatchNorm after output layer
        )

    def forward(self, x):
        return self.layers(x)


# Define the bottleneck
class BottleNeck(nn.Module):
    # creates symmetrical squeeze and unsqueeze
    def __init__(self, dimensions=[512, 128, 32]):
        super().__init__()
        squeeze_layers = []
        for i in range(len(dimensions) - 2):
            squeeze_layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            squeeze_layers.append(nn.BatchNorm1d(dimensions[i + 1]))
            squeeze_layers.append(nn.ReLU())
        squeeze_layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        squeeze_layers.append(nn.BatchNorm1d(dimensions[-1]))

        unsqueeze_layers = []
        inverse_dimension = list(dimensions)[::-1]
        for i in range(len(inverse_dimension) - 2):
            unsqueeze_layers.append(
                nn.Linear(inverse_dimension[i], inverse_dimension[i + 1])
            )
            unsqueeze_layers.append(nn.BatchNorm1d(inverse_dimension[i + 1]))
            unsqueeze_layers.append(nn.ReLU())
        unsqueeze_layers.append(nn.Linear(inverse_dimension[-2], inverse_dimension[-1]))
        unsqueeze_layers.append(nn.BatchNorm1d(inverse_dimension[-1]))

        self.squeeze_layers = nn.ModuleList(squeeze_layers)
        self.unsqueeze_layers = nn.ModuleList(unsqueeze_layers)

    def forward(self, x):
        """return both the squeezed feature and the unsqueezed result"""
        for l in self.squeeze_layers:
            x = l(x)
        feat = x.clone().detach()
        for l in self.unsqueeze_layers:
            x = l(x)
        return feat, x


# Integrate the feature extractor and projection head into one model
class ContrastiveModel(nn.Module):
    def __init__(self, base_model, projection_head, device=torch.device("cpu")):
        super(ContrastiveModel, self).__init__()
        self.base_model = base_model
        self.projection_head = projection_head
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        features = self.base_model(x)  # Extract base model features
        projections = self.projection_head(
            features
        )  # Project features for contrastive learning
        return features, projections


class ContrastiveResnet18(ContrastiveModel):
    def __init__(self, device=torch.device("cpu")):

        # Load a pretrained ResNet18 model
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # change the first layer to accept 1 channel input, by averaging the weights from conv1
        base_model.conv1 = cnn_architectures.change_conv2d_channels(
            base_model.conv1, num_channels=1
        )
        # Remove the final fully connected layer
        base_model.fc = nn.Identity()

        # create a 2-layer projection head
        projection_head = ProjectionHead(input_dim=512)  # 512 matches model output size

        super(ContrastiveResnet18, self).__init__(
            base_model, projection_head, device=device
        )


class ContrastiveResnet50(ContrastiveModel):
    def __init__(self, device=torch.device("cpu")):

        # Load a pretrained ResNet18 model
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # change the first layer to accept 1 channel input, by averaging the weights from conv1
        base_model.conv1 = cnn_architectures.change_conv2d_channels(
            base_model.conv1, num_channels=1
        )
        # Remove the final fully connected layer
        base_model.fc = nn.Identity()

        # create a 2-layer projection head
        projection_head = ProjectionHead(input_dim=2048)  # matches model output size

        super().__init__(base_model, projection_head, device=device)


class Resnet18_Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load a pretrained ResNet18 model
        self.embedder = resnet18_1ch_embedder()

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        emb = self.embedder(x)
        logits = self.classifier(emb)
        return emb, logits


class Resnet18_Embedder(nn.Module):
    """just the embedding model, but return two outputs to match API of other classes"""

    def __init__(self):
        super().__init__()
        self.embedder = resnet18_1ch_embedder()

    def forward(self, x):
        x = self.embedder(x)
        return (x, x)


class Resnet50_Embedder(nn.Module):
    """just the embedding model, but return two outputs to match API of other classes"""

    def __init__(self):
        super().__init__()
        self.embedder = resnet50_1ch_embedder()

    def forward(self, x):
        x = self.embedder(x)
        return (x, x)


def resnet18_1ch_embedder():
    # Load a pretrained ResNet18 model
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # change the first layer to accept 1 channel input, by averaging the weights from conv1
    m.conv1 = cnn_architectures.change_conv2d_channels(m.conv1, num_channels=1)
    # remove the final fully connected layer
    m.fc = nn.Identity()
    return m


class Resnet50_Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load a pretrained ResNet18 model
        self.embedder = resnet50_1ch_embedder()
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        emb = self.embedder(x)
        logits = self.classifier(emb)
        return emb, logits


def resnet50_1ch_embedder():
    # Load a pretrained ResNet18 model
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # change the first layer to accept 1 channel input, by averaging the weights from conv1
    m.conv1 = cnn_architectures.change_conv2d_channels(m.conv1, num_channels=1)
    # remove the final fully connected layer
    m.fc = nn.Identity()
    return m


class Resnet18_Bottleneck_Classifier(nn.Module):
    """forward returns (squeezed feature, logits)"""

    def __init__(self, num_classes, bottleneck_shape=[512, 128, 32]):
        super().__init__()
        # Load a pretrained ResNet18 model
        self.embedder = resnet18_1ch_embedder()
        self.bottleneck = BottleNeck(dimensions=bottleneck_shape)
        self.classifier = nn.Linear(bottleneck_shape[0], num_classes)

    def forward(self, x):
        x = self.embedder(x)
        feat, x = self.bottleneck(x)
        logits = self.classifier(x)
        return feat, logits

class HawkEarsOneModel(torch.nn.Module):
    """select one of the hawkears ensembled models, add separate classification head"""
    def __init__(self,num_classes=None):
        import bioacoustics_model_zoo as bmz

        super().__init__()
        # load largest ensembled hawkears pre-trained model
        self.embedder = bmz.HawkEars().network.models[4]
        # remove original classification head
        self.embedder.head.fc = torch.nn.Identity() 
        # create new classification head
        self.classifier = torch.nn.Linear(2048,num_classes)
    
    def forward(self,x):
        """returns (embeddings, logits)"""
        embeddings = self.embedder(x)
        logits = self.classifier(embeddings)
        return embeddings, logits
    
def hawkears_preprocessor():
    return bmz.HawkEars().preprocessor
    