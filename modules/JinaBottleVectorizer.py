# from transformers import AutoModel, AutoProcessor
# import numpy as np
# from PIL import Image

# class JinaBottleVectorizer:
#     def __init__(self, model_name="jinaai/jina-embeddings-v5-omni-small", default_task='clustering'):
#         if default_task not in ['retrieval', 'classification', 'clustering', 'text-matching']:
#             raise ValueError("default_task must be one of the folowing items: 'retrieval', 'classification', 'clustering', 'text-matching'")
#         self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, default_task=default_task).eval()
#         self.proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

#     def vectorize_array(self, crop: np.array):
#         i_vec = self.model.embed(
#             **self.proc(
#                 images=Image.fromarray(
#                     crop
#                 ), text="<|vision_start|><|image_pad|><|vision_end|>", return_tensors="pt"
#             ).to(self.model.device)
#         )
#         return i_vec

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor

class JinaBottleVectorizer:
    def __init__(
        self, 
        model_name="jinaai/jina-embeddings-v5-omni-small", 
        default_task='clustering'
    ):
        valid_tasks = ['retrieval', 'classification', 'clustering', 'text-matching']
        if default_task not in valid_tasks:
            raise ValueError(f"default_task must be one of: {valid_tasks}")
        
        print(f"Загрузка модели {model_name}...")
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            default_task=default_task,
            modality="vision"  # Только vision + text
        ).eval()
        
        self.proc = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True,
            modality="vision"
        )
        
        self.device = torch.device("cpu")
        self.model.to(self.device)
        print("Модель успешно загружена!")

    @torch.no_grad()
    def vectorize_array(self, crop: np.ndarray) -> list:
        """Векторизует numpy-массив (изображение) в плоский список для Qdrant."""
        if crop.ndim == 3 and crop.shape[-1] not in [3, 4]:
            raise ValueError(f"Ожидается изображение с 3 или 4 каналами, получено: {crop.shape}")
        
        # 1. Конвертация в PIL Image
        img = Image.fromarray(crop)
        
        # 2. Уменьшаем изображение до макс. 512x512 (критично для CPU и памяти)
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        # 3. Обработка процессором
        inputs = self.proc(
            images=img, 
            text="<image>", 
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        # 4. Получение эмбеддинга
        embedding_tensor = self.model.embed(**inputs)
        
        # 5. КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: добавляем .float() перед .numpy()
        # bfloat16 не поддерживается numpy напрямую, приводим к стандартному float32
        return embedding_tensor.cpu().float().numpy().reshape(-1).tolist()
# import torch
# import numpy as np
# from PIL import Image
# from sentence_transformers import SentenceTransformer

# class JinaBottleVectorizer:
#     def __init__(
#         self, 
#         model_name="jinaai/jina-embeddings-v5-omni-small", 
#         default_task='clustering'
#     ):
#         valid_tasks = ['retrieval', 'classification', 'clustering', 'text-matching']
#         if default_task not in valid_tasks:
#             raise ValueError(f"default_task must be one of: {valid_tasks}")
        
#         print(f"Загрузка модели {model_name}...")
#         self.model = SentenceTransformer(
#             model_name,
#             trust_remote_code=True,
#             model_kwargs={
#                 "default_task": default_task,
#                 "modality": "vision",  # Только vision + text
#                 "torch_dtype": torch.bfloat16  # Экономия памяти
#             }
#         )
        
#         self.device = torch.device("cpu")
#         print("Модель успешно загружена!")

#     # @torch.no_grad()
#     def vectorize_array(self, crop: np.ndarray) -> np.ndarray:
#         """Векторизует numpy-массив (изображение)."""
#         if crop.ndim == 3 and crop.shape[-1] not in [3, 4]:
#             raise ValueError(f"Ожидается изображение с 3 или 4 каналами, получено: {crop.shape}")
        
#         img = Image.fromarray(crop)
        
#         # sentence-transformers автоматически обрабатывает изображения
#         embedding = self.model.encode(
#             img,
#             convert_to_numpy=True,
#             normalize_embeddings=True  # L2 нормализация
#         )
        
#         return embedding

# # Проверка
# if __name__ == "__main__":
#     vectorizer = JinaBottleVectorizer(default_task='clustering')
#     dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
#     vector = vectorizer.vectorize_array(dummy_image)
#     print(f"Размерность вектора: {vector.shape}")
#     print(f"Первые 5 элементов: {vector[:5]}")