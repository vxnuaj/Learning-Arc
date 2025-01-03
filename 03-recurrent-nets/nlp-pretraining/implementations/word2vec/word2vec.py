from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

class LossCallback(CallbackAny2Vec):
    def __init__(self, model):
        self.losses = []
        self.model = model
        self.epochs = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        epoch = len(self.losses)
        self.epochs.append(epoch)
        
        print(f"Epoch {epoch}: Loss = {loss}")
        

    def save_model(self):
        self.model.save("word2vec_model.model")
        print("Model saved to 'word2vec_model.model'")

def load_data(path):
    sequences = []
    with open(path, 'r') as f:
        for sequence in f:
            sequence = sequence.split()
            sequences.append(sequence)
    return sequences

sequences = load_data('data/trainv2.txt')

# Hyperparameters
embed_dim = 100
window_size = 5
epochs = 500

print("Initializing Model")
model = Word2Vec(
    sentences=sequences,
    vector_size=embed_dim,  
    window=window_size,     
    min_count=2,            
    sg=0,                   
    epochs=epochs,
    compute_loss = True
)

print('building vocab')
model.build_vocab(sequences)
print(f"Model initialized with {len(model.wv.index_to_key)} unique words")

loss_callback = LossCallback(model)

print("Training")
model.train(sequences, total_examples=model.corpus_count, epochs=epochs, callbacks=[loss_callback])

loss_callback.save_model()
