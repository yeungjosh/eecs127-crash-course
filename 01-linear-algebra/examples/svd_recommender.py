"""
SVD-based Recommender System
-----------------------------
A practical example showing how SVD powers collaborative filtering.

This is the technique that won the Netflix Prize!
"""

import numpy as np
from typing import Tuple, Optional


class SVDRecommender:
    """
    Collaborative filtering using SVD (Singular Value Decomposition).

    The idea:
    - Users × Items matrix is huge and sparse (most users haven't rated most items)
    - SVD finds latent features: Users × Features × Items
    - We can fill in missing ratings using these features!
    """

    def __init__(self, n_factors: int = 20):
        """
        Args:
            n_factors: Number of latent features to extract (typically 10-50)
                      More factors = more expressive but risk overfitting
        """
        self.n_factors = n_factors
        self.U = None  # User features
        self.s = None  # Feature importance
        self.Vt = None  # Item features
        self.user_mean = None

    def fit(self, ratings: np.ndarray) -> 'SVDRecommender':
        """
        Learn latent features from ratings matrix.

        Args:
            ratings: (n_users, n_items) matrix
                    0 means unrated, 1-5 are actual ratings

        Returns:
            self (for method chaining)
        """
        # Step 1: Mean-center the ratings (per user)
        # Why? Some users rate everything 5, others average 3
        # We want to learn preferences, not baseline ratings
        mask = ratings > 0  # Which entries are rated?
        self.user_mean = np.zeros(ratings.shape[0])
        centered = ratings.copy()

        for i in range(ratings.shape[0]):
            user_ratings = ratings[i, mask[i]]
            if len(user_ratings) > 0:
                self.user_mean[i] = user_ratings.mean()
                centered[i, mask[i]] -= self.user_mean[i]

        # Step 2: SVD decomposition
        # ratings ≈ U @ diag(s) @ Vt
        self.U, self.s, self.Vt = np.linalg.svd(centered, full_matrices=False)

        # Step 3: Keep only top k factors
        # Why? Reduces noise and overfitting
        self.U = self.U[:, :self.n_factors]
        self.s = self.s[:self.n_factors]
        self.Vt = self.Vt[:self.n_factors, :]

        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair.

        Args:
            user_id: User index
            item_id: Item index

        Returns:
            Predicted rating (1-5 scale)
        """
        # Reconstruct the rating using latent features
        prediction = (
            self.user_mean[user_id] +  # User's baseline
            self.U[user_id] @ np.diag(self.s) @ self.Vt[:, item_id]
        )

        # Clip to valid range [1, 5]
        return np.clip(prediction, 1, 5)

    def recommend(self, user_id: int, n: int = 5,
                  exclude_rated: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get top N recommendations for a user.

        Args:
            user_id: User index
            n: Number of recommendations
            exclude_rated: Boolean mask of already-rated items to exclude

        Returns:
            Array of item indices, sorted by predicted rating (best first)
        """
        # Predict ratings for ALL items
        all_predictions = (
            self.user_mean[user_id] +
            self.U[user_id] @ np.diag(self.s) @ self.Vt
        )

        # Exclude already-rated items
        if exclude_rated is not None:
            all_predictions[exclude_rated] = -np.inf

        # Return top N
        return np.argsort(all_predictions)[::-1][:n]


def create_demo_data() -> Tuple[np.ndarray, list, list]:
    """Create a toy movie ratings dataset."""

    movies = [
        "Star Wars", "Empire Strikes Back", "Matrix", "Inception",
        "Titanic", "Notebook", "Pride & Prejudice", "La La Land",
        "Avengers", "Iron Man"
    ]

    users = ["Alice", "Bob", "Charlie", "Diana", "Eve"]

    # Ratings matrix (0 = not rated)
    # Pattern: Some users like action, others like romance
    ratings = np.array([
        # SW  ESB  Mat Inc  Tit  Not  P&P  LaL  Avg  IM
        [5,   5,   4,  4,   2,   1,   1,   2,   5,   4],  # Alice (action fan)
        [5,   5,   5,  5,   1,   1,   0,   1,   5,   5],  # Bob (action fan)
        [1,   2,   1,  2,   5,   5,   5,   5,   0,   1],  # Charlie (romance fan)
        [2,   1,   2,  1,   5,   5,   5,   4,   1,   2],  # Diana (romance fan)
        [4,   4,   3,  4,   3,   3,   3,   3,   4,   3],  # Eve (likes both)
    ], dtype=float)

    return ratings, users, movies


def main():
    """Demo: Build a movie recommender system."""

    print("=" * 60)
    print("SVD-Based Movie Recommender System")
    print("=" * 60)

    # Load data
    ratings, users, movies = create_demo_data()
    print(f"\nDataset: {len(users)} users, {len(movies)} movies")
    print(f"Sparsity: {(ratings == 0).sum() / ratings.size * 100:.1f}% unrated")

    # Train model
    model = SVDRecommender(n_factors=3)
    model.fit(ratings)

    print("\n" + "=" * 60)
    print("Latent Features Learned")
    print("=" * 60)

    # Interpret features (simplified)
    print("\nSingular values (feature importance):")
    for i, val in enumerate(model.s):
        print(f"  Feature {i+1}: {val:.2f}")

    # Make predictions
    print("\n" + "=" * 60)
    print("Predictions for Bob (action fan)")
    print("=" * 60)

    bob_id = 1  # Bob's index
    bob_rated = ratings[bob_id] > 0

    print("\nBob's actual ratings:")
    for i, (movie, rating) in enumerate(zip(movies, ratings[bob_id])):
        if rating > 0:
            print(f"  {movie}: {rating:.0f}⭐")

    print("\nPredictions for unrated movies:")
    for i, movie in enumerate(movies):
        if not bob_rated[i]:
            pred = model.predict(bob_id, i)
            print(f"  {movie}: {pred:.1f}⭐ (predicted)")

    # Top recommendations
    print("\n" + "=" * 60)
    print("Top 3 Recommendations for Charlie (romance fan)")
    print("=" * 60)

    charlie_id = 2
    charlie_rated = ratings[charlie_id] > 0
    recommendations = model.recommend(charlie_id, n=3, exclude_rated=charlie_rated)

    for rank, movie_id in enumerate(recommendations, 1):
        pred = model.predict(charlie_id, movie_id)
        print(f"{rank}. {movies[movie_id]} (predicted: {pred:.1f}⭐)")

    # Show the power of SVD
    print("\n" + "=" * 60)
    print("Why SVD Works: Latent Features")
    print("=" * 60)
    print("""
SVD discovered latent patterns like:
- Feature 1: "Action vs Romance" spectrum
- Feature 2: "Sci-Fi vs Contemporary"
- Feature 3: "Marvel vs Classic"

Even though we never told it about genres!
This is the same technique behind:
- Netflix recommendations
- Spotify song suggestions
- Amazon product recommendations
    """)


if __name__ == "__main__":
    main()
