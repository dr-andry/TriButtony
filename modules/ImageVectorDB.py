from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

class ImageVectorDB:
    def __init__(
        self, 
        url: str = "http://localhost:6333", 
        collection_name: str = "images_embeddings",
        vector_size: int = 1024, # По умолчанию для jina-embeddings-v5-omni-small
        distance: models.Distance = models.Distance.COSINE
    ):
        """
        Инициализация клиента Qdrant и создание коллекции, если её нет.
        """
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Проверяет наличие коллекции и создает её при необходимости."""
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=self.distance,
                ),
                # Опционально: можно включить квантование для экономии RAM (ScalarQuantization)
                # quantization_config=models.ScalarQuantization(
                #     scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
                # )
            )
            print(f"Коллекция '{self.collection_name}' успешно создана.")
        else:
            print(f"Коллекция '{self.collection_name}' уже существует.")

    def upsert_images(self, image_names: List[str], vectors: List[List[float]]) -> bool:
        """
        Добавляет или обновляет вектора и имена изображений в Qdrant.
        
        :param image_names: Список строк с именами/путями к изображениям.
        :param vectors: Список векторов (списков float), соответствующих изображениям.
        :return: True при успешном добавлении.
        """
        if len(image_names) != len(vectors):
            raise ValueError("Количество имен изображений и векторов должно совпадать.")

        points = [
            models.PointStruct(
                id=str(uuid.uuid4()), # В реальном проекте лучше использовать хэш имени или UUID
                vector=vector,
                payload={"image_name": name}
            )
            for idx, (name, vector) in enumerate(zip(image_names, vectors))
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        return True

    def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Ищет наиболее похожие изображения по заданному вектору.
        
        :param query_vector: Вектор запроса.
        :param limit: Количество возвращаемых результатов.
        :return: Список словарей с именем изображения и оценкой сходства (score).
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        return [
            {"image_name": hit.payload["image_name"], "score": hit.score}
            for hit in results
        ]