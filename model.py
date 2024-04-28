import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
from pytorch_lightning import Trainer


# Define a dataset class to handle data loading and batching
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, description = self.data[idx]
        return {"query": query, "description": description}


# Define a PyTorch Lightning DataModule for organizing data loading logic
class SimpleDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=2):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


# Define the model class using Sentence Transformers framework
class SBERTEmbedder(pl.LightningModule):
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.devices = 1

    def forward(self, input_texts):
        # Generate embeddings for input texts
        embeddings = self.model.encode(input_texts, convert_to_tensor=True, show_progress_bar=False)
        return embeddings.requires_grad_()

    def training_step(self, batch, batch_idx):
        # Perform a forward pass and calculate loss
        query_embeddings = self.forward(batch['query'])
        description_embeddings = self.forward(batch['description'])
        loss = torch.nn.functional.mse_loss(query_embeddings, description_embeddings)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        # Define optimizer
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    def train_model(self, data_module, max_epochs=1):
        trainer = Trainer(max_epochs=max_epochs, accelerator=self.accelerator, devices=self.devices)
        trainer.fit(self, data_module)


# Example data with a few query-description pairs
mock_data = [
    ("high end water bottle",
     "[TITLE] Simple Modern 40 oz Tumbler with Handle and Straw Lid | Insulated Cup Reusable Stainless Steel Water Bottle Travel Mug Cupholder Friendly | Gifts for Women Men Him Her | Trek Collection | Almond Birch [COLOR] purple [BRAND] Zojirushi [PRODUCT_TYPE] DRINKING_CUP [ITEM_TYPE_KEYWORD] tumbler"),
    ("durable travel mug",
     "[TITLE] Contigo AUTOSEAL West Loop Stainless Steel Travel Mug | Spill-proof Coffee Mug with Easy-Clean Lid | Fits Car Cup Holders | 16 oz [COLOR] black [BRAND] Contigo [PRODUCT_TYPE] DRINKING_CUP [ITEM_TYPE_KEYWORD] travel mug"),
    ("smart home speaker",
     "[TITLE] Echo Dot (3rd Gen) | Smart speaker with Alexa | Charcoal Fabric [COLOR] charcoal [BRAND] Amazon [PRODUCT_TYPE] SMART_HOME [ITEM_TYPE_KEYWORD] smart speaker"),
    ("ergonomic office chair",
     "[TITLE] Gabrylly Ergonomic Mesh Office Chair, High Back Desk Chair - Adjustable Headrest with Flip-Up Arms, Tilt Function, Lumbar Support and PU Wheels, Swivel Computer Task Chair [COLOR] black [BRAND] Gabrylly [PRODUCT_TYPE] OFFICE_FURNITURE [ITEM_TYPE_KEYWORD] office chair"),
    ("compact wireless keyboard",
     "[TITLE] Logitech K380 Multi-Device Bluetooth Keyboard | Windows, Mac, Chrome OS, Android, iPad, iPhone, Apple TV Compatible | with Flow Cross-Computer Control and Easy-Switch up to 3 Devices [COLOR] blue [BRAND] Logitech [PRODUCT_TYPE] COMPUTER_ACCESSORY [ITEM_TYPE_KEYWORD] keyboard")
]

# Setup dataset and data module
dataset = SimpleDataset(mock_data)
data_module = SimpleDataModule(dataset)

# Create the model and train
model = SBERTEmbedder()
model.train_model(data_module)

# Post-training: Evaluate the model and print embeddings for the first few samples
model.eval()
sample_data = ["high end water bottle", "smart home speaker", "compact wireless keyboard"]
with torch.no_grad():
    sample_embeddings = model(sample_data)


torch.set_printoptions(precision=3, threshold=5000, edgeitems=3, linewidth=150, sci_mode=False)
torch.save(sample_embeddings, 'sample_embeddings.pt')

embeddings = torch.load('sample_embeddings.pt')
print(embeddings)
