import numpy as np
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch
import re
from typing import Dict, List, Tuple, Set, Optional
import numpy.typing as npt

class DeterministicLanguageModel:
    def __init__(
        self,
        latent_dim: int = 768,
        epsilon: float = 0.1,
        initial_radius: float = 0.5,
        min_radius: float = 0.1,
        max_radius: float = 0.9
    ):
        """
        Initialize the deterministic language learning system.
        
        Args:
            latent_dim: Dimension of the unified latent space
            epsilon: Error tolerance for confidence certification
            initial_radius: Starting radius for new certified regions
            min_radius: Minimum allowed confidence radius
            max_radius: Maximum allowed confidence radius
        """
        # Initialize the unified latent space parameters
        self.latent_dim = latent_dim
        self.epsilon = epsilon
        self.initial_radius = initial_radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        
        # Initialize language models and tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        
        # Initialize certified knowledge regions
        self.certified_points: List[Dict] = []  # List of certified experience points
        self.global_confidence_region: float = 0.0  # Measure of total certified region
        
        # Initialize dimensionality reduction for visualization
        self.pca = PCA(n_components=2)
        
        # TF-IDF vectorizer for additional text features
        self.tfidf = TfidfVectorizer(max_features=1000)
        
    def embed_experience(
        self,
        observation: str,
        action: str
    ) -> npt.NDArray[np.float32]:
        """
        Map an (observation, action) pair to the unified latent space Z.
        Uses a combination of BERT embeddings and TF-IDF features.
        
        Args:
            observation: Input text from user
            action: Response text
            
        Returns:
            Latent space embedding
        """
        # Get BERT embeddings
        obs_encoding = self._get_bert_embedding(observation)
        act_encoding = self._get_bert_embedding(action)
        
        # Get TF-IDF features
        combined_text = f"{observation} [SEP] {action}"
        tfidf_features = self._get_tfidf_features(combined_text)
        
        # Combine all features
        combined_features = np.concatenate([
            obs_encoding,
            act_encoding,
            tfidf_features
        ])
        
        # Project to latent dimension if necessary
        if combined_features.shape[0] != self.latent_dim:
            combined_features = self._project_to_latent_space(combined_features)
            
        return combined_features
        
    def compute_confidence_radius(
        self,
        z: npt.NDArray[np.float32],
        certified_points: List[Dict]
    ) -> float:
        """
        Compute the confidence radius around point z where model behavior
        is guaranteed to be within epsilon tolerance.
        
        Uses binary search to find largest valid radius.
        """
        low = 0.0
        high = self.max_radius
        
        while high - low > 1e-4:
            mid = (low + high) / 2
            if self._check_confidence_ball(z, mid, certified_points):
                low = mid
            else:
                high = mid
                
        return low
        
    def _check_confidence_ball(
        self,
        z: npt.NDArray[np.float32],
        radius: float,
        certified_points: List[Dict]
    ) -> bool:
        """
        Check if all points within radius of z have responses
        within epsilon tolerance.
        """
        # Get all certified points within the radius
        for point in certified_points:
            dist = np.linalg.norm(z - point['embedding'])
            if dist <= radius:
                response_diff = self._response_distance(
                    point['response'],
                    self._get_response(z)
                )
                if response_diff > self.epsilon:
                    return False
        return True
        
    def _response_distance(
        self,
        response1: str,
        response2: str
    ) -> float:
        """
        Compute semantic distance between two responses.
        Uses combination of embedding distance and linguistic features.
        """
        # Get embeddings
        emb1 = self._get_bert_embedding(response1)
        emb2 = self._get_bert_embedding(response2)
        
        # Compute cosine distance
        cos_dist = 1 - np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
        
        # Add linguistic feature distances
        ling_dist = self._linguistic_distance(response1, response2)
        
        return 0.7 * cos_dist + 0.3 * ling_dist
        
    def _linguistic_distance(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute linguistic distance based on syntax, semantics, and discourse features.
        """
        # Implement sophisticated linguistic distance metrics
        # This is a simplified version
        features1 = self._extract_linguistic_features(text1)
        features2 = self._extract_linguistic_features(text2)
        
        return np.linalg.norm(features1 - features2)
        
    def update_certified_regions(
        self,
        new_point: Dict
    ) -> None:
        """
        Update certified regions when new experience is added.
        Merges overlapping regions and updates confidence radii.
        """
        # Add new point
        self.certified_points.append(new_point)
        
        # Update radii of nearby points
        self._update_nearby_radii(new_point)
        
        # Merge overlapping regions
        self._merge_regions()
        
        # Update global confidence measure
        self._update_global_confidence()
        
    def get_response(
        self,
        observation: str
    ) -> Tuple[Optional[str], float]:
        """
        Generate response for new observation.
        Returns response and confidence radius.
        """
        # Embed observation
        z = self.embed_experience(observation, "")
        
        # Find closest certified point
        closest_point = None
        min_dist = float('inf')
        
        for point in self.certified_points:
            dist = np.linalg.norm(z - point['embedding'])
            if dist < min_dist:
                min_dist = dist
                closest_point = point
                
        if closest_point is None:
            return None, 0.0
            
        # Check if within confidence radius
        if min_dist <= closest_point['radius']:
            return closest_point['response'], closest_point['radius']
        
        return None, 0.0
        
    def learn(
        self,
        observation: str,
        correct_response: str
    ) -> None:
        """
        Learn from new experience by updating certified regions.
        """
        # Create new certified point
        z = self.embed_experience(observation, correct_response)
        
        new_point = {
            'embedding': z,
            'observation': observation,
            'response': correct_response,
            'radius': self.initial_radius
        }
        
        # Update certified regions
        self.update_certified_regions(new_point)
        
    def _get_bert_embedding(self, text: str) -> npt.NDArray[np.float32]:
        """Get BERT embedding for text."""
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.encoder(**tokens)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
    def _get_tfidf_features(self, text: str) -> npt.NDArray[np.float32]:
        """Get TF-IDF features for text."""
        try:
            return self.tfidf.transform([text]).toarray().squeeze()
        except:
            self.tfidf.fit([text])
            return self.tfidf.transform([text]).toarray().squeeze()
            
    def _project_to_latent_space(
        self,
        features: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Project features to latent dimension."""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        if features.shape[1] > self.latent_dim:
            return PCA(n_components=self.latent_dim).fit_transform(features).squeeze()
        return features.squeeze()

    def _update_nearby_radii(self, new_point: Dict) -> None:
        """Update confidence radii of points near new experience."""
        for point in self.certified_points:
            dist = np.linalg.norm(new_point['embedding'] - point['embedding'])
            if dist <= point['radius'] + new_point['radius']:
                point['radius'] = self.compute_confidence_radius(
                    point['embedding'],
                    self.certified_points
                )

    def _merge_regions(self) -> None:
        """Merge overlapping certified regions."""
        # Implement sophisticated region merging
        # This is a placeholder for the full implementation
        pass

    def _update_global_confidence(self) -> None:
        """Update measure of global confidence region."""
        # Implement sophisticated volume computation
        # This is a placeholder for the full implementation
        total_volume = sum(point['radius']**self.latent_dim for point in self.certified_points)
        self.global_confidence_region = total_volume

def run_interactive_session():
    """Run interactive learning session."""
    model = DeterministicLanguageModel()
    print("Starting interactive learning session...")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break
            
        response, confidence = model.get_response(user_input)
        
        if response is None:
            print("Agent: I'm not certified to respond to this input.")
            correct_response = input("Please provide the correct response: ").strip()
            model.learn(user_input, correct_response)
            print(f"Agent: Thank you! I've learned this response with initial radius {model.initial_radius}")
        else:
            print(f"Agent: {response}")
            print(f"Confidence radius: {confidence:.4f}")
            feedback = input("Was this response appropriate? (yes/no): ").strip().lower()
            
            if feedback != 'yes':
                correct_response = input("Please provide the correct response: ").strip()
                model.learn(user_input, correct_response)
                print("Agent: Thank you for the correction!")

if __name__ == "__main__":
    run_interactive_session()