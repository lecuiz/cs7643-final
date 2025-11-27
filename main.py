from final_project.src.rnn import ImageCaptioningModel as rnn_model
from final_project.src.train import train
from final_project.src.utils import Config, ArtemisDataset
from torchvision import transforms, models

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

dataset = ArtemisDataset(Config.PICKLE_PATH, transform=transform, split='train')
dataset.df = dataset.df.sample(1000)

model = rnn_model(
    vocab_size=dataset.vocab_size,
    embed_dim=Config.EMBED_DIM,
    hidden_dim=Config.HIDDEN_DIM,
    # num_layers=Config.NUM_LAYERS,
    dropout=Config.DROPOUT
)

train(model=model, dataset=dataset)
