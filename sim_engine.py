def validate_similarity_matrix(similarity_matrix, paper_ids, tolerance=1e-6):
    """
    Validate the similarity matrix properties
    
    Parameters:
    -----------
    similarity_matrix : numpy.ndarray
        Similarity matrix to validate
    paper_ids : list
        List of paper IDs corresponding to matrix rows/columns
    tolerance : float, optional
        Tolerance for floating-point comparisons
        
    Returns:
    --------
    dict
        Validation results with 'is_valid' flag and list of 'issues'
    """
    import numpy as np
    import logging
    
    logger = logging.getLogger(__name__)
    
    validation = {
        'is_valid': True,
        'issues': []
    }
    
    # Check dimensions
    if similarity_matrix.shape[0] != len(paper_ids) or similarity_matrix.shape[1] != len(paper_ids):
        validation['is_valid'] = False
        validation['issues'].append(f"Matrix dimensions {similarity_matrix.shape} don't match paper count {len(paper_ids)}")
    
    # Check if symmetric
    if not np.allclose(similarity_matrix, similarity_matrix.T, atol=tolerance):
        validation['is_valid'] = False
        validation['issues'].append("Matrix is not symmetric")
    
    # Check diagonal values (should be 1.0)
    if not np.allclose(np.diag(similarity_matrix), 1.0, atol=tolerance):
        validation['is_valid'] = False
        validation['issues'].append("Diagonal values are not all 1.0")
    
    # Check for negative values
    if np.any(similarity_matrix < 0):
        validation['is_valid'] = False
        validation['issues'].append("Matrix contains negative values")
    
    # Check for NaN values
    if np.any(np.isnan(similarity_matrix)):
        validation['is_valid'] = False
        validation['issues'].append("Matrix contains NaN values")
    
    # Log validation results
    if validation['is_valid']:
        logger.info("Similarity matrix passed all validation checks")
    else:
        logger.warning(f"Similarity matrix validation failed with issues: {validation['issues']}")
    
    return validation

def calculate_fragmentation_metrics(similarity_matrix, paper_ids, entity_graph, entity_relationship_key):
    """
    Calculate fragmentation metrics for an entity type (author, journal, JEL code)
    
    Parameters:
    -----------
    similarity_matrix : numpy.ndarray
        Similarity matrix
    paper_ids : list
        List of paper IDs corresponding to matrix rows/columns
    entity_graph : dict
        Entity relationship graph from build_entity_relationship_graph
    entity_relationship_key : str
        Key for the entity-to-paper relationship in the graph (e.g., 'author_to_paper')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with fragmentation metrics for each entity
    """
    import numpy as np
    import pandas as pd
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Verify the relationship exists in the graph
    if entity_relationship_key not in entity_graph['relationships']:
        logger.error(f"Relationship '{entity_relationship_key}' not found in the entity graph")
        return pd.DataFrame()
    
    # Get entity to papers mapping
    entity_to_papers = entity_graph['relationships'][entity_relationship_key]
    
    # Create paper ID to index mapping
    paper_id_to_idx = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}
    
    # Prepare results container
    results = []
    
    # Count entities with multiple papers
    multi_paper_entities = 0
    valid_entities = 0
    
    # Process each entity
    for entity_id, papers in entity_to_papers.items():
        # Skip entities with fewer than 2 papers
        if len(papers) < 2:
            continue
            
        multi_paper_entities += 1
            
        # Get indices for this entity's papers that are in our similarity matrix
        valid_papers = [p for p in papers if p in paper_id_to_idx]
        
        # Skip if we don't have enough valid papers
        if len(valid_papers) < 2:
            continue
            
        valid_entities += 1
        
        paper_indices = [paper_id_to_idx[p] for p in valid_papers]
        
        # Extract similarity submatrix for this entity's papers
        entity_sim_matrix = similarity_matrix[np.ix_(paper_indices, paper_indices)]
        
        # Calculate metrics (only upper triangle, excluding diagonal)
        sim_values = entity_sim_matrix[np.triu_indices_from(entity_sim_matrix, k=1)]
        
        # Calculate statistics
        mean_similarity = np.mean(sim_values)
        median_similarity = np.median(sim_values)
        max_similarity = np.max(sim_values)
        min_similarity = np.min(sim_values)
        
        # Compute fragmentation index (inverse of mean similarity)
        # Add smoothing factor to avoid division by zero
        epsilon = 1e-5
        fragmentation_index = 1.0 / (mean_similarity + epsilon)
        
        # Store results
        results.append({
            'entity_id': entity_id,
            'paper_count': len(valid_papers),
            'mean_similarity': mean_similarity,
            'median_similarity': median_similarity,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
            'fragmentation_index': fragmentation_index
        })
    
    logger.info(f"Found {multi_paper_entities} entities with multiple papers")
    logger.info(f"Analyzed {valid_entities} entities with multiple papers in the similarity matrix")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    if len(df_results) > 0:
        # Sort by fragmentation index (descending)
        df_results = df_results.sort_values('fragmentation_index', ascending=False)
        
        logger.info(f"Fragmentation metrics results:")
        logger.info(f"  Total entities analyzed: {len(df_results)}")
        logger.info(f"  Mean fragmentation index: {df_results['fragmentation_index'].mean():.4f}")
        logger.info(f"  Mean within-entity similarity: {df_results['mean_similarity'].mean():.4f}")
    else:
        logger.warning(f"No fragmentation metrics could be calculated")
    
    return df_results