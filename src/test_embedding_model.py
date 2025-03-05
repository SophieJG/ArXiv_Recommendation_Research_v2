from collections import defaultdict
import glob
import gzip
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import copy
import ast
import math
from util import authors_path, data_dir, kaggle_data_path, papers_path, tmp_data_dir
from models.spector_embed import Specter2Embedder

def multi_file_query(
    files_path: str,
    processing_func: callable,
    n_jobs: int,
    *args,
    **kwargs
):
    """
This is a utility function facilitating parallel load and processing of data from disk. All files 
in `files_path` are processed by applying `processing_func` on them. The number of files processed
in parallel is given by `n_jobs`. *args and **kwargs are passed to `processing_func`

Arguments:
    files_path: the path to the list of files to load and process. We run glob(files_path) to get the list
        of files
    processing_func: the function used to process the chunk files
    n_jobs: how many processes to run in parallel
    *args: passed to processing_func
    **kwargs: passed to processing_func

Returns:
    a list containing the outputs of processing_func
"""
    files = glob.glob(files_path)
    return Parallel(n_jobs=n_jobs)(delayed(processing_func)(f, *args, **kwargs) for f in tqdm(files, f"n_jobs={n_jobs}"))



def _generate_paper_embeddings_inner(paper_id_abstract_dict: dict, n_jobs: int):
    """
    A helper function to generate embeddings for a single paper. Should only be called through process_paper_embedding.
    Takes a paper ID and returns a dictionary with the paper ID and its generated embedding.

    Arguments:
        paper_id: string ID of the paper to generate embedding for
        paper_info: dictionary containing paper information including abstracts
    Returns:
        Dictionary mapping paper ID to its embedding vector
    """

    def _generate_single_embedding(pid, abstract):
        print(f"Generating embedding for paper {pid}")
        paper_embeddings = {}
        try:
            # Initialize embedder
            embedder = Specter2Embedder()
                
            # Generate embedding
            embedding = embedder.compute_embedding(abstract)
            
            # Store result
            paper_embeddings[str(pid)] = embedding.tolist()
            
        except Exception as e:
            print(f"Error generating embedding for paper {pid}: {str(e)}")
            
        return paper_embeddings

    # Convert dictionary items to list and then chunk
    chunk_size = math.ceil(len(paper_id_abstract_dict) / n_jobs)
    items = list(paper_id_abstract_dict.items())
    chunks = [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]

    # Process chunks in parallel
    results = Parallel(n_jobs=-1)(
        delayed(_generate_single_embedding)(pid, abstract) 
        for chunk in tqdm(chunks, desc="Generating embeddings") 
        for pid, abstract in chunk.items()
    )

    # Merge results
    final_embeddings = {}
    for r in results:
        final_embeddings.update(r)

    return final_embeddings

def test_embedding_model():
    """
    Using the list of corpus ids of the Arxiv papers we query the `embedding` table to get spector2 embedding for each paper.
    """
    # multi_file_query returns a list of dicts. Merge it to a single dict. Note that a single paper can have citations in multiple
    # res files so the dictionaries needs to be "deep" merged
    
    missing_paperid_abstract_dict = {156053035: "Learning with noisy labels is one of the hottest problems in weakly-supervised learning. Based on memorization effects of deep neural networks, training on small-loss instances becomes very promising for handling noisy labels. This fosters the state-of-the-art approach \u201cCo-teaching\u201d that cross-trains two deep neural networks using the small-loss trick. However, with the increase of epochs, two networks converge to a consensus and Co-teaching reduces to the self-training MentorNet. To tackle this issue, we propose a robust learning paradigm called Co-teaching+, which bridges the \u201cUpdate by Disagreement\u201d strategy with the original Co-teaching. First, two networks feed forward and predict all data, but keep prediction disagreement data only. Then, among such disagreement data, each network selects its small-loss data, but back propagates the small-loss data from its peer network and updates its own parameters. Empirical results on benchmark datasets demonstrate that Coteaching+ is much superior to many state-of-theart methods in the robustness of trained models.",
    208857791: "In this work we introduce the DP-auto-GAN framework for synthetic data generation, which combines the low dimensional representation of autoencoders with the flexibility of Generative Adversarial Networks (GANs). This framework can be used to take in raw sensitive data, and privately train a model for generating synthetic data that will satisfy the same statistical properties as the original data. This learned model can be used to generate arbitrary amounts of publicly available synthetic data, which can then be freely shared due to the post-processing guarantees of differential privacy. Our framework is applicable to unlabeled mixed-type data, that may include binary, categorical, and real-valued data. We implement this framework on both unlabeled binary data (MIMIC-III) and unlabeled mixed-type data (ADULT). We also introduce new metrics for evaluating the quality of synthetic mixed-type data, particularly in unsupervised settings.",
    195833706: "Vulnerabilities in third-party libraries is a growing concern for the software developer, as it poses risks not only to the software client itself but to the entire software ecosystem. To mitigate these risks, developers are strongly recommended to update their dependencies. Recent studies show that affected developers are not likely to respond to the vulnerability threat. However, another reason for the lag of vulnerability updates is due to slow repackaging (i.e., package the vulnerability fix into a new version) and delivery (i.e., affected client adopt the new version) of the fix. To understand these lags of updates, we use both qualitative and quantitative approaches to conduct an empirical study on how 188 fixes were repackaged and delivered across over eight hundred thousand releases of npm software clients hosted on GitHub. We report two lags: (1) lags in repackaging occur as vulnerability fixes are more likely to be bundled with other non-related updates (i.e., about 83.33\\% of commits are not related to the fix) and (2) lags in the delivery are caused by clients that are more likely to adopt the minor fix than adopt the patch fix. Furthermore, other factors such as downstream dependencies and severity do have an impact. We also find that freshness of packages does not impact the amount of lags. The identification of these two lags opens up different avenues on how to facilitate faster fix delivery throughout a library ecosystem.",
    247058622: "We introduce the task of retrieving relevant video moments from a large corpus of untrimmed, unsegmented videos given a natural language query. Our task poses unique challenges as a system must efficiently identify both the relevant videos and localize the relevant moments in the videos. To address these challenges, we propose SpatioTemporal Alignment with Language (STAL), a model that represents a video moment as a set of regions within a series of short video clips and aligns a natural language query to the moment's regions. Our alignment cost compares variable-length language and video features using symmetric squared Chamfer distance, which allows for efficient indexing and retrieval of the video moments. Moreover, aligning language features to regions within a video moment allows for finer alignment compared to methods that extract only an aggregate feature from the entire video moment. We evaluate our approach on two recently proposed datasets for temporal localization of moments in video with natural language (DiDeMo and Charades-STA) extended to our video corpus moment retrieval setting. We show that our STAL re-ranking model outperforms the recently proposed Moment Context Network on all criteria across all datasets on our proposed task, obtaining relative gains of 37% - 118% for average recall and up to 30% for median rank. Moreover, our approach achieves more than 130x faster retrieval and 8x smaller index size with a 1M video corpus in an approximate setting.",
}
    missing_paper_count = 5

    embedding_papers = {}

    if missing_paper_count > 0:
        print(f"Generating embeddings for {missing_paper_count} papers")
        missing_paper_embeddings = _generate_paper_embeddings_inner(missing_paperid_abstract_dict, 3)
        embedding_papers.update(missing_paper_embeddings)
    
    print(f"embedding_papers:\n", embedding_papers)

if __name__ == "__main__":
    test_embedding_model()