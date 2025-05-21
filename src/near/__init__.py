try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss-gpu-cuvs is required but not installed. "
        "Please install it with: "
        "conda install -c pytorch -c nvidia -c rapidsai faiss-gpu-cuvs=1.11.0"
    )
