# Vector embedding demos

A vector embedding encodes an input as a list of floating point numbers.

"dog" → [0.017198, -0.007493, -0.057982, 0.054051, -0.028336, 0.019245,…]

Different models output different embeddings, with varying lengths.

| Model | Encodes | Vector length |
| --- | --- | --- |
| word2vec | words | 300 |
| Sbert (Sentence-Transformers) | text (up to ~400 words) | 768 |
| OpenAI text-embedding-ada-002 | text (up to 8191 tokens) | 1536 |
| OpenAI text-embedding-3-small | text (up to 8191 tokens) | 256-1536 |
| OpenAI text-embedding-3-large | text (up to 8191 tokens) | 256-3072 |
| Azure AI Vision | image or text | 1024 |

Vector embeddings are commonly used for similarity search, fraud detection, recommendation systems, and RAG (Retrieval-Augmented Generation).

This repository contains a visual exploration of vectors, using several embedding models.

## Getting started

Before running the notebooks, install the requirements:

```shell
pip install -r requirements.txt
```

Then explore these notebooks:

* [Generate new OpenAI text embeddings](generate_embedding.ipynb)
* [Compare OpenAI and Word2Vec embeddings](comparison.ipynb)
* [Vector similarity](similarity.ipynb)
* [Vector search](search.ipynb)
* [Generate multimodal vectors for dataset](prep_multimodal.ipynb)
* [Explore multimodal vectors](multimodal_vectors.ipynb)
* [Vector distance metrics](distance_metrics.ipynb)
* [Vector quantization](quantization.ipynb)
* [Vector dimension reduction (MRL)](dimension_reduction.ipynb)

These notebooks are also provided, but aren't necessary unless you're generating new embeddings data.

* [Generate OpenAI embeddings for datasets](prep_openai.ipynb)
* [Generate Word2Vec embeddings for datasets](prep_word2vec_gnews.ipynb)

## Deploying Azure OpenAI embedding models

If you need to generate new OpenAI embeddings, you'll need access to the embedding models via the API. This project includes infrastructure as code (IaC) to provision an Azure OpenAI deployment of "text-embedding-3-large". The IaC is defined in the `infra` directory and uses the Azure Developer CLI to provision the resources.

1. Make sure the [Azure Developer CLI (azd)](https://aka.ms/install-azd) is installed.

2. Login to Azure:

    ```shell
    azd auth login
    ```

    For GitHub Codespaces users, if the previous command fails, try:

   ```shell
    azd auth login --use-device-code
    ```

3. Provision the OpenAI account:

    ```shell
    azd provision
    ```

    It will prompt you to provide an `azd` environment name (like "vector-demos"), select a subscription from your Azure account, and select a location. Then it will provision the resources in your account.

4. Once the resources are provisioned, you should now see a local `.env` file with all the environment variables needed to run the scripts.
5. To delete the resources, run:

    ```shell
    azd down
    ```

## Resources

Each notebook has resources at the bottom to dig further into that topic. Here are some additional general resources:

* [Poster: Visual exploration of vectors (PyCon 2025)](https://tinyurl.com/visual-vectors-poster)
* [Livestream Recording: Vector Embeddings](https://www.youtube.com/watch?v=lyIJcHKA6b0) from [Python + AI Series](https://techcommunity.microsoft.com/blog/EducatorDeveloperBlog/learn-python--ai-from-our-video-series/4400393)
* [Vector compression techniques](https://github.com/microsoft/rag-time/tree/main/Journey%203%20-%20Optimize%20your%20Vector%20Index%20for%20Scale)
