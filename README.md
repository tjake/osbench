# OpenSearch Parquet Loader

This script efficiently loads data from Parquet files into an OpenSearch index, leveraging batch processing and multiple connections to maximize indexing speed.

## Prerequisites

- Python 3.8+
- Access to an OpenSearch cluster

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/osbench.git
    cd osbench
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure environment variables:**
    Create a `.env` file in the project root and add your OpenSearch connection details:
    ```
    OPENSEARCH_HOSTS='["http://localhost:9200"]'
    OPENSEARCH_INDEX="my-index"
    ```

## Usage

Run the script from the command line, providing the path to your Parquet file and the number of parallel workers:

```bash
python osbench.py /path/to/your/data.parquet --workers 4
```

-   `/path/to/your/data.parquet`: The Parquet file to load.
-   `--workers`: The number of parallel connections to use.
