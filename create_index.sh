curl -X PUT "https://localhost:9200/jvector-index?pretty" --insecure -H 'Content-Type: application/json' -d'
{
  "settings": {
    "index": {
      "knn": true
    }
  },
  "mappings": {
    "_source": {
                "excludes": ["embeddings", "chunk_id"],
                "recovery_source_excludes": ["embeddings", "chunk_id"]
     },
    "properties": {
			"chunk_id": {"type": "long"},
      "embeddings": {
        "type": "knn_vector",
        "method": {
          "name": "disk_ann",
          "space_type": "l2",
          "engine": "jvector",
          "parameters": {
            "m": 32,
            "ef_construction": 200
          }
        },
        "dimension": 384
      }
    }
  }
}' -u admin
