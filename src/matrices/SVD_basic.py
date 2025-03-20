from matrices.base_matrix_model import BaseMatrixModel
import os
from citation_matrix_data import load_citation_matrix
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from typing import List, Tuple

class SVDBasicModel(BaseMatrixModel):
    """
    SVD model for matrix factorization. This class is not meant to be used directly, but rather as a base class for other matrix models.
    """
    def __init__(self, params: dict):
        self.matrix = None
        self.node_list = None
        self.node_to_index = None # Mapping from node id to index in the matrix

        self.n_components = params.get("n_components", 128)  # Number of latent dimensions
        self.n_iter = params.get("n_iter", 10)  # Number of iterations for SVD

        self.svd = TruncatedSVD(n_components=self.n_components, n_iter=self.n_iter, random_state=42)
        self.node_embeddings = None  # Placeholder for node embeddings

        self.update_matrix_when_new_papers = params.get("update_matrix_when_new_papers", False)  # Whether to update the matrix with new papers
        self.refit_when_new_papers = params.get("refit_when_new_papers", False)  # Whether to refit the model when new papers are added

    def load_matrix(self, matrix_path) -> None:
        self.matrix, self.node_list, self.node_to_index = load_citation_matrix(matrix_path)

    def fit(
        self,
        train_samples: list = None,
        validation_samples: list = None
    ):
        """
        Perform matrix factorization using SVD.
        """
        assert train_samples is None, "train_samples must be None for matrix models"
        assert validation_samples is None, "validation_samples must be None for matrix models"
        assert self.matrix is not None, "Matrix must be loaded before fitting"
        assert self.node_list is not None, "Node list must be loaded before fitting"
        assert self.node_to_index is not None, "Node to index mapping must be loaded before fitting"

        # The TruncatedSVD model computes a factorization M ≈ U Σ Vᵀ.
        # The transform() method returns the low-dimensional representation (U*Σ).
        print("Fitting SVD model...")
        self.svd.fit(self.matrix)
        print("SVD fitting complete.")

        # Get low-dimensional embeddings for each paper.
        # Here, each row of node_embeddings corresponds to U*Σ.
        self.node_embeddings = self.svd.transform(self.matrix)
        print("Node embeddings shape:", self.node_embeddings.shape)


    @staticmethod
    def _process_samples(samples: list) -> List[Tuple[Tuple[int, List[int]], List[Tuple[int, List[int]]]]]:
        paper_author_pairs = []
        for sample in samples:
            paper_refs = sample["paper_id"], sample.get("references", []), sample.get("update_date", None)
            author_paper_refs = [(p["paper_id"], p.get("references", []), p["year"], p.get("publicationdate", None)) for p in sample["author"]["papers"]]
            # sort author_paper_refs by publication date (earliest first)
            author_paper_refs.sort(key=lambda x: (x[2] if x[2] is not None else float('inf'), 
                                                pd.Timestamp(x[3]) if x[3] is not None else pd.Timestamp.max), 
                                 reverse=False)

            paper_author_pairs.append((paper_refs, author_paper_refs))
        
        # sort paper_author_pairs by paper_refs update_date date (earliest first)
        paper_author_pairs.sort(key=lambda x: (x[0][2] if x[0][2] is not None else pd.Timestamp.max), reverse=False)

        # remove the date information
        paper_author_pairs = [((paper_refs[0], paper_refs[1]), [(p[0], p[1]) for p in author_paper_refs]) for paper_refs, author_paper_refs in paper_author_pairs]
        
        return paper_author_pairs
    
    def _update_matrix_with_new_papers(self, new_papers_with_refs: list):
        """
        Update the citation matrix with new papers and their references.
        Args:
            new_papers_with_refs (list): List of tuples where each tuple contains a new paper ID and its references.
                                        Example: [(new_paper_id, [ref1, ref2, ...]), ...]
        """
        # Create a new row for each new paper
        new_rows = []
        for paper_id, refs in new_papers_with_refs:
            new_row = np.zeros(len(self.node_list))
            for ref_id in refs:
                if ref_id in self.node_to_index:
                    new_row[self.node_to_index[ref_id]] = 1
                # Ignore references that are not already in the matrix
            if paper_id not in self.node_to_index:
                # Add new paper to node_list and node_to_index
                new_index = len(self.node_list)
                self.node_list.append(paper_id)
                self.node_to_index[paper_id] = new_index
                # Expand the matrix with a new column for the new paper
                self.matrix = sp.hstack([self.matrix, sp.csr_matrix((self.matrix.shape[0], 1))])
            new_rows.append(new_row)
        
        # Convert new rows to sparse matrix and append to existing matrix
        new_rows_sparse = sp.csr_matrix(new_rows)
        self.matrix = sp.vstack([self.matrix, new_rows_sparse])

    def _embed_new_paper(self, new_paper_vector):
        """
        Given a new paper’s citation vector (as a sparse row vector of shape (1, n_nodes)),
        use the fitted SVD model to compute its low-dimensional embedding.
        
        In this example, new_paper_vector should indicate which training papers 
        (columns) the new paper cites (e.g., 1 for a citation, 0 otherwise).
        
        Returns:
            A 1D numpy array of length latent_dim representing the embedding.
        """
        # We can use the SVD model’s transform() to “fold in” the new data.
        # (Note: TruncatedSVD.transform() does the same projection as was applied to M.)
        return self.svd.transform(new_paper_vector)[0]
    
    @staticmethod
    def cosine_similarity(a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b, axis=1)
        return (a.dot(b.T) / (a_norm * b_norm + 1e-10))

    # TODO: test this out
    def predict_proba(self, samples: list):
        """
        Run inference on a list of samples
        """
        predictions = []

        paper_author_pairs = self._process_samples(samples)
        i = 0
        end = len(paper_author_pairs)
        for paper_refs, author_paper_refs in paper_author_pairs:
            if i % (end // 20) == 0:
                print(f"Calculating Predictions for paper {i}/{end}")
            i += 1

            new_row = np.zeros(len(self.node_list))
            for paper_id in paper_refs[1]:
                if paper_id in self.node_to_index:
                    new_row[self.node_to_index[paper_id]] = 1
                # else:
                #     print(f"Warning: paper {paper_id} is not in matrix.")

            new_row_sparse = sp.csr_matrix(new_row)
            new_embedding = self._embed_new_paper(new_row_sparse)

            # Get embeddings for author's previous papers
            author_embeddings = []
            for paper_id, refs in author_paper_refs:
                if paper_id in self.node_to_index:
                    author_idx = self.node_to_index[paper_id]
                    author_embeddings.append(self.node_embeddings[author_idx])
                else:
                    # Handle new author paper by creating its embedding
                    new_author_row = np.zeros(len(self.node_list))
                    for ref_id in refs:
                        if ref_id in self.node_to_index:
                            new_author_row[self.node_to_index[ref_id]] = 1
                    new_author_row_sparse = sp.csr_matrix(new_author_row)
                    new_author_embedding = self._embed_new_paper(new_author_row_sparse)
                    author_embeddings.append(new_author_embedding)

            if author_embeddings:
                author_embeddings = np.vstack(author_embeddings)
                # Calculate similarity between new paper and author's papers
                similarities = self.cosine_similarity(new_embedding, author_embeddings)
                # Convert cosine similarity from [-1,1] to [0,1] range and take max
                prediction = (np.max(similarities) + 1) / 2
            else:
                prediction = 0.0

            predictions.append(prediction)

        if self.update_matrix_when_new_papers:
            for paper_refs, author_paper_refs in paper_author_pairs:
                new_papers_with_refs = [(p[0], p[1]) for p in author_paper_refs] + [(paper_refs[0], paper_refs[1])]
                self._update_matrix_with_new_papers(new_papers_with_refs)
                if self.refit_when_new_papers:
                    self.fit()  # Re-fit the SVD model with the updated matrix

        return np.array(predictions)
    
    def predict_proba_ranking(self, papers: list, authors: list):
        """
        Run inference on the cartesian product between all papers and all authors.
        Returns a numpy array of shape (n_authors, n_papers) with similarity scores.
        """
        # Precompute paper embeddings
        paper_embeddings = []
        for paper in papers:
            paper_id = paper.get("paper_id")
            if paper_id in self.node_to_index:
                # Use existing embedding if paper is already in the matrix
                idx = self.node_to_index[paper_id]
                emb = self.node_embeddings[idx]
            else:
                # Create new embedding from references if paper is not in matrix
                row = np.zeros(len(self.node_list))
                for ref in paper.get("references", []):
                    if ref in self.node_to_index:
                        row[self.node_to_index[ref]] = 1
                row_sparse = sp.csr_matrix(row)
                emb = self._embed_new_paper(row_sparse)
            paper_embeddings.append(emb)
        
        # Compute ranking scores for each author over all papers
        utility = np.zeros((len(authors), len(papers)))
        for i, author in tqdm(enumerate(authors), "Calculating Predictions for each author"):
            author_embs = []
            # Process the author's papers
            for p in author.get("author", {}).get("papers", []):
                pid = p.get("paper_id")
                if pid in self.node_to_index:
                    idx = self.node_to_index[pid]
                    author_embs.append(self.node_embeddings[idx])
                else:
                    row = np.zeros(len(self.node_list))
                    for ref in p.get("references", []):
                        if ref in self.node_to_index:
                            row[self.node_to_index[ref]] = 1
                    row_sparse = sp.csr_matrix(row)
                    emb = self._embed_new_paper(row_sparse)
                    author_embs.append(emb)
            if author_embs:
                author_embs = np.vstack(author_embs)
                for j, paper_emb in enumerate(paper_embeddings):
                    sim = self.cosine_similarity(paper_emb, author_embs)
                    # check that sim shape is correct
                    assert sim.shape == (len(author_embs),), f"Expected shape {(len(author_embs),)}, but got {sim.shape}"
                    # Convert cosine similarity from [-1,1] to [0,1] range
                    utility[i, j] = (np.max(sim) + 1) / 2  # scale to [0,1]
            else:
                print("OH NO! NO paper embeddings for author ", author)
                utility[i, :] = 0.0
        return utility

    def _save(self, path: str):
        """
        Save the SVD model and related data to the specified path.
        """
        import joblib
        os.makedirs(path, exist_ok=True)
        # Save SVD model object
        joblib.dump(self.svd, os.path.join(path, "svd_model.pkl"))
        # Save matrix data and mappings
        joblib.dump({
            "matrix": self.matrix,
            "node_list": self.node_list,
            "node_to_index": self.node_to_index,
            "node_embeddings": self.node_embeddings
        }, os.path.join(path, "svd_data.pkl"))
    
    def _load(self, path: str):
        """
        Load the SVD model and related data from the specified path.
        """
        import joblib
        # Load SVD model object
        self.svd = joblib.load(os.path.join(path, "svd_model.pkl"))
        # Load matrix data and mappings
        data = joblib.load(os.path.join(path, "svd_data.pkl"))
        self.matrix = data["matrix"]
        self.node_list = data["node_list"]
        self.node_to_index = data["node_to_index"]
        self.node_embeddings = data["node_embeddings"]

