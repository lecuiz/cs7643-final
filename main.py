from final_project.src.train import train, test
from final_project.src.utils import Config, ArtemisDataset
from final_project.src.inference import run_inference
from torchvision import transforms
from final_project.src.decoders.decoder_rnn import ImageCaptioningModel as rnn_model
from final_project.src.decoders_w_simple_attention.decoder_lstm import ImageCaptioningModel as lstm_model
from final_project.src.decoders_w_multihead_attention.decoder_transformer import \
    ImageCaptioningModel as transformer_model
from final_project.src.decoders_w_multihead_attention.decoder_rnn import ImageCaptioningModel as attention_rnn_model
from final_project.src.decoders_w_multihead_attention.decoder_lstm import ImageCaptioningModel as attention_lstm_model

# from final_project.src.decoders.decoder_rnn import ImageCaptioningModel as vanilla_rnn_model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

dataset = ArtemisDataset(Config.MINI_PICKLE_PATH, transform=transform, split='train')

model = rnn_model(
    vocab_size=Config.VOCAB_SIZE,
    embed_dim=Config.EMBED_DIM,
    hidden_dim=Config.HIDDEN_DIM,
    dropout=Config.DROPOUT
)

# model = transformer_model(
#     vocab_size=dataset.vocab_size,
#     embed_dim=Config.EMBED_DIM,
#     hidden_dim=Config.HIDDEN_DIM,
#     num_layers=Config.NUM_LAYERS,
#     num_heads=Config.NUM_HEADS,
#     dropout=Config.DROPOUT
# )

# model = attention_lstm_model(
#     vocab_size=dataset.vocab_size,
#     embed_dim=Config.EMBED_DIM,
#     hidden_dim=Config.HIDDEN_DIM,
#     num_heads=Config.NUM_HEADS,
#     dropout=Config.DROPOUT
# )

train(model=model, dataset=dataset)
test(model=model, dataset=dataset)
run_inference(model=model, dataset=dataset, use_pregen_vocab=True)
