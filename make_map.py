from absl import flags, app
import os 
import numpy as np
import json

from sklearn.manifold import TSNE
import umap

FLAGS = flags.FLAGS
flags.DEFINE_string('embeddings_file', 
                     default=None, 
                     required=True, 
                     help="Path to multi-dimensional embeddings' JSON")
flags.DEFINE_string('out_folder', 
                    default=None, 
                    required=True, 
                    help='Path to save map')
flags.DEFINE_integer('dim', 
                    default=2, 
                    required=False, 
                    help='Dimensionality of the map (2 for the traditional map, 3 for a volume)')
flags.DEFINE_string('strategy', 
                    default='tsne', 
                    required=False, 
                    help="Dimensionality reduction strategy (for now, must be either 'tsne' or 'umap')")


def main(argv):
    with open(FLAGS.embeddings_file, 'rb') as f:
        embeddings = json.load(f)
    all_embeddings = np.concatenate(list(embeddings.values()))
    
    if FLAGS.strategy=='tsne':
        tsne = TSNE(n_components=FLAGS.dim)
        embeddings_2d = tsne.fit_transform(all_embeddings)
    elif FLAGS.strategy=='umap':
        umap_transform = umap.UMAP(n_components=FLAGS.dim).fit(embeddings)
        embeddings_2d = umap_transform.embedding_    
    else:
        raise NotImplementedError(f'Strategy {FLAGS.strategy} has not been implemented')
    embeddings_2d = {filename: emb_2d.tolist() for filename, emb_2d in zip(embeddings.keys(), embeddings_2d)}
    
    os.makedirs(FLAGS.out_folder, exist_ok=True)
    file_name = os.path.join(FLAGS.out_folder, f'{FLAGS.strategy}_embeddings_{FLAGS.dim}d.json')
    with open(file_name, 'w') as f:
        json.dump(embeddings_2d, f)
        

if __name__=='__main__':
    app.run(main)