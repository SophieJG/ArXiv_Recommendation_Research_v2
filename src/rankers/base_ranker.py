import pandas as pd
import numpy as np


class BaseRanker:
    def __init__(
        self,
        items_to_rank: int,
        params: dict
    ):
        pass

    def rank(
        self,
        utility: pd.DataFrame,
        paper_embeddings: dict
    ):
        raise NotImplementedError("rank for BaseRanker must be overloaded")

    def rank_with_sampled_negatives(
        self,
        utility: pd.DataFrame,
        author_to_positive_papers: dict,
        num_negatives: int,
        paper_embeddings: dict = None
    ):
        """
        Rank papers for each author using one randomly sampled positive paper and
        a specified number of randomly sampled negative papers.
        
        Args:
            utility: DataFrame with authors as index and papers as columns
            author_to_positive_papers: Dict mapping author IDs to sets of positive paper IDs
            num_negatives: Number of negative papers to sample for each author
            paper_embeddings: Optional dict of paper embeddings
            
        Returns:
            Tuple containing:
            - Dict mapping author IDs to ranked lists of paper IDs
            - Dict mapping author IDs to their sampled positive paper ID
        """
        rng = np.random.default_rng(seed=42)
        local_ranked = {}
        sampled_positives = {}  # Store the sampled positive paper for each author
        
        # Convert utility index and columns to strings for consistent keying
        utility.index = utility.index.map(str)
        utility.columns = utility.columns.map(str)
        
        # Initialize all_papers after converting columns to strings
        all_papers = set(utility.columns)
        
        for author in utility.index:
            # Skip authors with no positive papers
            if author not in author_to_positive_papers or not author_to_positive_papers[author]:
                continue
                
            # Sample one positive paper randomly
            positive_papers = list(author_to_positive_papers[author])
            sampled_positive = rng.choice(positive_papers)
            sampled_positives[author] = sampled_positive  # Store the sampled positive
            
            # Sample random negatives (papers not in positive papers for this author)
            available_negatives = list(all_papers - set(author_to_positive_papers[author]))
            
            if len(available_negatives) >= num_negatives:
                sampled_negatives = rng.choice(available_negatives, size=num_negatives, replace=False)
            else:
                sampled_negatives = np.array(available_negatives)
            
            # Create a subset utility matrix with just the sampled papers
            sampled_papers = [sampled_positive] + list(sampled_negatives)
            local_utility = utility.loc[[author], sampled_papers]
            
            # Use the ranker's regular rank method to rank these sampled papers
            local_ranked[author] = self.rank(local_utility, paper_embeddings)[author]
            
        return local_ranked, sampled_positives
