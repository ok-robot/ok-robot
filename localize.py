from abc import ABC, abstractmethod

class Localizer(ABC):
    @abstractmethod
    def load_pretrained(self):
        pass

    @abstractmethod
    def calculate_clip_and_st_embeddings_for_queries(self, queries):
        pass
    
    @abstractmethod
    def find_alignment_over_model(self, queries):
        pass

    # Currently we only support compute one query each time, in the future we might want to support check many queries

    @abstractmethod
    def localize_AonB(self, A, B, k_A, k_B,data_type):
        pass

    @abstractmethod
    def find_alignment_for_A(self, A):
        pass
