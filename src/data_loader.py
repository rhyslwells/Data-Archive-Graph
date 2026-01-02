"""
data_loader.py
Module for loading and filtering graph data from JSON files.
"""

import json
from typing import List, Dict, Any, Optional


def load_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        List of node dictionaries
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_by_tag(data: List[Dict[str, Any]], tag: str) -> List[Dict[str, Any]]:
    """
    Filter nodes that have the specified tag.
    
    Args:
        data: List of node dictionaries
        tag: Tag to filter by
        
    Returns:
        Filtered list of nodes containing the tag
    """
    return [item for item in data if tag in (item.get('tags') or [])]


def filter_by_titles(data: List[Dict[str, Any]], titles: List[str]) -> List[Dict[str, Any]]:
    """
    Filter nodes by a list of titles (normalized filenames).
    
    Args:
        data: List of node dictionaries
        titles: List of normalized filenames to include
        
    Returns:
        Filtered list of nodes matching the titles
    """
    title_set = set(titles)
    return [item for item in data if item['normalized_filename'] in title_set]


def filter_by_category(data: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
    """
    Filter nodes by category.
    
    Args:
        data: List of node dictionaries
        category: Category to filter by (e.g., 'ML', 'CS')
        
    Returns:
        Filtered list of nodes in the category
    """
    return [item for item in data if item.get('category') == category]


def get_all_tags(data: List[Dict[str, Any]]) -> set:
    """
    Get all unique tags from the dataset.
    
    Args:
        data: List of node dictionaries
        
    Returns:
        Set of all unique tags
    """
    all_tags = set()
    for item in data:
        tags = item.get('tags')
        if tags:
            all_tags.update(tags)
    return all_tags


def get_all_categories(data: List[Dict[str, Any]]) -> set:
    """
    Get all unique categories from the dataset.
    
    Args:
        data: List of node dictionaries
        
    Returns:
        Set of all unique categories
    """
    return {item.get('category') for item in data if item.get('category')}


def print_data_summary(data: List[Dict[str, Any]]) -> None:
    """
    Print a summary of the dataset.
    
    Args:
        data: List of node dictionaries
    """
    print(f"\n=== Data Summary ===")
    print(f"Total nodes: {len(data)}")
    print(f"Categories: {sorted(get_all_categories(data))}")
    print(f"Number of unique tags: {len(get_all_tags(data))}")
    print(f"Available tags: {sorted(get_all_tags(data))}")


# Example usage
if __name__ == '__main__':
    # Load data
    data_path = '../data/1-categories_snapshot_linked.json'
    raw_data = load_data(data_path)
    
    # Print summary
    print_data_summary(raw_data)
    
    # Example: Filter by tag
    filtered_by_tag = filter_by_tag(raw_data, 'analysis')
    print(f"\nNodes with 'analysis' tag: {len(filtered_by_tag)}")
    
    # Example: Filter by titles
    titles_to_find = ['data_analysis', 'eda', 'data_visualisation']
    filtered_by_titles = filter_by_titles(raw_data, titles_to_find)
    print(f"Nodes matching titles: {len(filtered_by_titles)}")
    
    # Example: Filter by category
    filtered_by_category = filter_by_category(raw_data, 'ML')
    print(f"Nodes in ML category: {len(filtered_by_category)}")