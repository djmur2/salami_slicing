"""
Metadata integration for salami-slicing analysis.
Handles mapping between papers, authors, journals, JEL codes and other metadata.
"""

def standardize_field_names(df, standard_mapping=None):
    """
    Standardize column names across different data files
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to standardize column names
    standard_mapping : dict, optional
        Mapping from current names to standard names
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with standardized column names
    """
    if standard_mapping is None:
        # Default mappings for common field variations
        standard_mapping = {
            'paper': 'paper_id',
            'author_user': 'author_id',
            'name': 'author_name',
            'jel': 'jel_code',
            'published_text': 'publication_info'
        }
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Apply mappings
    for old_name, new_name in standard_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    return df

def build_entity_relationship_graph(data_files):
    """
    Build a comprehensive graph of relationships between entities
    (papers, authors, JEL codes, journals, etc.)
    
    Parameters:
    -----------
    data_files : dict
        Dictionary mapping entity type to file path
        e.g. {'author': 'data/raw/author_user.tsv', 'jel': 'data/raw/jel.tsv'}
        
    Returns:
    --------
    dict
        Dictionary of entity-to-entity mappings
    """
    import pandas as pd
    import logging
    from collections import defaultdict
    
    logger = logging.getLogger(__name__)
    
    # Initialize the graph structure
    graph = {
        'relationships': {}
    }
    
    # Process author relationships
    if 'author' in data_files and data_files['author']:
        try:
            logger.info(f"Building author relationships from {data_files['author']}")
            author_df = pd.read_csv(data_files['author'], sep='\t', encoding='utf-8')
            
            # Standardize column names
            if 'paper' in author_df.columns and 'paper_id' not in author_df.columns:
                author_df = author_df.rename(columns={'paper': 'paper_id'})
            
            # Ensure required columns exist
            if 'paper_id' not in author_df.columns:
                logger.error(f"Required column 'paper_id' not found in {data_files['author']}")
            elif 'author_user' not in author_df.columns:
                logger.error(f"Required column 'author_user' not found in {data_files['author']}")
            else:
                # Drop missing values
                initial_count = len(author_df)
                author_df = author_df.dropna(subset=['paper_id', 'author_user'])
                dropped_count = initial_count - len(author_df)
                if dropped_count > 0:
                    logger.warning(f"Dropped {dropped_count} rows with missing values in author data")
                
                # Build paper -> authors mapping
                paper_to_authors = defaultdict(list)
                author_to_papers = defaultdict(list)
                
                for _, row in author_df.iterrows():
                    paper_id = row['paper_id']
                    author_id = row['author_user']
                    paper_to_authors[paper_id].append(author_id)
                    author_to_papers[author_id].append(paper_id)
                
                # Convert to regular dicts
                graph['relationships']['paper_to_author'] = dict(paper_to_authors)
                graph['relationships']['author_to_paper'] = dict(author_to_papers)
                
                logger.info(f"Built author relationships: {len(paper_to_authors)} papers, {len(author_to_papers)} authors")
        except Exception as e:
            logger.error(f"Error building author relationships: {str(e)}")
    
    # Process JEL code relationships
    if 'jel' in data_files and data_files['jel']:
        try:
            logger.info(f"Building JEL code relationships from {data_files['jel']}")
            jel_df = pd.read_csv(data_files['jel'], sep='\t', encoding='utf-8')
            
            # Standardize column names
            if 'paper' in jel_df.columns and 'paper_id' not in jel_df.columns:
                jel_df = jel_df.rename(columns={'paper': 'paper_id'})
            
            # Ensure required columns exist
            if 'paper_id' not in jel_df.columns:
                logger.error(f"Required column 'paper_id' not found in {data_files['jel']}")
            elif 'jel' not in jel_df.columns:
                logger.error(f"Required column 'jel' not found in {data_files['jel']}")
            else:
                # Build paper -> JEL codes mapping
                paper_to_jels = defaultdict(list)
                jel_to_papers = defaultdict(list)
                
                for _, row in jel_df.iterrows():
                    paper_id = row['paper_id']
                    jel_code = row['jel']
                    paper_to_jels[paper_id].append(jel_code)
                    jel_to_papers[jel_code].append(paper_id)
                
                # Convert to regular dicts
                graph['relationships']['paper_to_jel'] = dict(paper_to_jels)
                graph['relationships']['jel_to_paper'] = dict(jel_to_papers)
                
                logger.info(f"Built JEL relationships: {len(paper_to_jels)} papers, {len(jel_to_papers)} JEL codes")
        except Exception as e:
            logger.error(f"Error building JEL relationships: {str(e)}")
    
    # Process publication information for journal extraction
    if 'published' in data_files and data_files['published']:
        try:
            logger.info(f"Extracting journal information from {data_files['published']}")
            published_df = pd.read_csv(data_files['published'], sep='\t', encoding='utf-8')
            
            # Standardize column names
            if 'paper' in published_df.columns and 'paper_id' not in published_df.columns:
                published_df = published_df.rename(columns={'paper': 'paper_id'})
            
            # Ensure required columns exist
            if 'paper_id' not in published_df.columns:
                logger.error(f"Required column 'paper_id' not found in {data_files['published']}")
            elif 'published_text' not in published_df.columns:
                logger.error(f"Required column 'published_text' not found in {data_files['published']}")
            else:
                # Drop missing values
                initial_count = len(published_df)
                published_df = published_df.dropna(subset=['paper_id', 'published_text'])
                dropped_count = initial_count - len(published_df)
                if dropped_count > 0:
                    logger.warning(f"Dropped {dropped_count} rows with missing publication information")
                
                # Extract journal information using simple pattern matching
                import re
                
                paper_to_journal = {}
                journal_to_papers = defaultdict(list)
                
                # Common journal patterns
                journal_patterns = [
                    r'"([^"]+)"',  # Text in quotes
                    r'in\s+([^,\.]+)',  # Text after "in" before comma or period
                    r'([^,]+),\s*[Vv]ol',  # Text before "vol." or "Vol."
                ]
                
                for _, row in published_df.iterrows():
                    paper_id = row['paper_id']
                    pub_text = row['published_text']
                    
                    if not isinstance(pub_text, str) or len(pub_text) < 5:
                        continue
                    
                    # Try each pattern to extract journal
                    journal = None
                    for pattern in journal_patterns:
                        match = re.search(pattern, pub_text)
                        if match:
                            journal = match.group(1).strip()
                            break
                    
                    if journal:
                        paper_to_journal[paper_id] = journal
                        journal_to_papers[journal].append(paper_id)
                
                # Add to graph
                graph['relationships']['paper_to_journal'] = paper_to_journal
                graph['relationships']['journal_to_paper'] = dict(journal_to_papers)
                
                logger.info(f"Extracted journal information for {len(paper_to_journal)} papers, found {len(journal_to_papers)} journals")
        except Exception as e:
            logger.error(f"Error extracting journal information: {str(e)}")
    
    return graph

def extract_analysis_features(graph, paper_ids=None):
    """
    Extract features for regression analysis from the entity relationship graph
    
    Parameters:
    -----------
    graph : dict
        Entity relationship graph from build_entity_relationship_graph
    paper_ids : list, optional
        List of paper IDs to include in analysis
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with features for each paper
    """
    import pandas as pd
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Initialize feature DataFrame
    if paper_ids is None:
        # If no paper_ids provided, use all papers in the graph
        paper_relationships = [
            rel for rel in graph['relationships'].keys() 
            if rel.startswith('paper_to_') and rel.endswith('_forward')
        ]
        
        if not paper_relationships:
            logger.error("No paper relationships found in the graph")
            return pd.DataFrame()
            
        # Use the first relationship to get paper IDs
        rel_name = paper_relationships[0]
        paper_ids = list(graph['relationships'][rel_name].keys())
    
    # Create DataFrame with paper_id as index
    features = pd.DataFrame(index=paper_ids)
    features.index.name = 'paper_id'
    
    # Add features
    
    # 1. Author-related features
    if 'paper_to_author_forward' in graph['relationships']:
        paper_to_authors = graph['relationships']['paper_to_author_forward']
        author_to_papers = graph['relationships']['paper_to_author_backward']
        
        # Number of authors per paper
        features['author_count'] = [
            len(paper_to_authors.get(pid, [])) for pid in paper_ids
        ]
        
        # Average papers per author (productivity measure)
        features['avg_author_papers'] = [
            sum(len(author_to_papers.get(aid, [])) 
                for aid in paper_to_authors.get(pid, [])) / max(1, len(paper_to_authors.get(pid, [])))
            for pid in paper_ids
        ]
    
    # 2. JEL-related features
    if 'paper_to_jel_forward' in graph['relationships']:
        paper_to_jels = graph['relationships']['paper_to_jel_forward']
        
        # Number of JEL codes per paper
        features['jel_count'] = [
            len(paper_to_jels.get(pid, [])) for pid in paper_ids
        ]
        
        # Has JEL code flag
        features['has_jel'] = features['jel_count'] > 0
    
    # 3. Journal-related features
    if 'paper_to_journal_forward' in graph['relationships']:
        paper_to_journals = graph['relationships']['paper_to_journal_forward']
        
        # Has journal information flag
        features['has_journal'] = [
            len(paper_to_journals.get(pid, [])) > 0 for pid in paper_ids
        ]
    
    # Convert boolean columns to integers for regression
    for col in features.select_dtypes(include=['bool']).columns:
        features[col] = features[col].astype(int)
    
    logger.info(f"Extracted {len(features.columns)} features for {len(paper_ids)} papers")
    
    return features.reset_index()