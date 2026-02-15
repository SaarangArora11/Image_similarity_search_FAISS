# Image Similarity Search with FAISS & ResNet50

A high-performance "Reverse Image Search" engine that retrieves visually similar images from a dataset.

This project combines Deep Learning for feature extraction with Facebook AI Similarity Search (FAISS) for efficient vector indexing. Instead of matching images by pixel values (which is brittle), it matches them by semantic meaning encoded in high-dimensional vector space.
## Objective

The goal of this project was to build a scalable recommendation system for visual data. It demonstrates how to:

  1. Convert unstructured image data into structured vector embeddings.
  2. Perform efficient similarity searches that scale to millions of items (unlike standard brute-force matching).
  3. Leverage Transfer Learning to understand image content without training a model from scratch.

## Key Concepts & Skills

  * Vector Embeddings: Representing complex images as fixed-size vectors of numbers.
  * Transfer Learning (ResNet50): Using a pre-trained Convolutional Neural Network (CNN) as a feature extractor.
  * Approximate Nearest Neighbors (ANN): Understanding how libraries like FAISS optimize search speed over pure accuracy.
  * Euclidean Distance (L2): The mathematical metric used to determine "similarity" in the embedding space.

## Methodology / Architecture
1. Feature Extraction (The "Brain")

I used ResNet50, pre-trained on ImageNet, as the backbone.

  * Input: Images resized to 224×224.
  * Process: The top classification layer (Softmax) is removed. We capture the output of the global average pooling layer.
  * Output: A 2048-dimensional vector representing the "essence" of the image (shapes, textures, colors).

2. Vector Indexing (The "Engine")

I used FAISS (Facebook AI Similarity Search) to index these vectors.
Index Type used: IndexFlatL2 (Exact Euclidean search).
This structure allows for rapid retrieval of the k-nearest neighbors for any query vector.

3. Inference Pipeline

    Step 1: Load a query image.

    Step 2: Pass it through ResNet50 to get its vector.

    Step 3: Query the FAISS index to find the 5 closest vectors.

    Step 4: Return and display the corresponding images from the dataset.

## Code Highlight

Here is how I extracted embeddings and built the FAISS index:
```Python

# Feature Extraction
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).flatten()

# Building the Index
d = 2048 # ResNet50 output dimension
index = faiss.IndexFlatL2(d) 
index.add(feature_vectors) # Add all image vectors to the database
```

## Results

The system successfully identifies visually similar items, even when they differ in angle or lighting.

  * Query: A specific sneaker.
  * Result: Retrieves 5 other sneakers of similar shape and color from the dataset.
  * Performance: Query time is in milliseconds due to FAISS optimization.

(Note: While IndexFlatL2 is fast for thousands of images, for millions of images, an IVF (Inverted File) index would be used for faster approximate search).

## How to Run

  1. Clone the repository.
  2. Install dependencies:
```    Bash

    pip install tensorflow faiss-cpu matplotlib
```
3. Add your dataset images to the images/ folder (or update the path in the notebook).
4. Run vectordb-for-images-using-faiss.ipynb.

## Future Improvements

* Implement IVF (Inverted File Index) to speed up search on massive datasets.
* Deploy the model as a REST API using FastAPI.
* Experiment with Vision Transformers (ViT) for potentially better feature extraction.

## References

FAISS Documentation
ResNet50 Paper
