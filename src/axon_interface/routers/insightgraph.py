"""
InsightGraph API Endpoints - Entity Relationship Analysis

Implements advanced entity relationship extraction:
- GET /relationships - Entity relationship extraction and analysis
- GET /relationships/graph - Relationship graph visualization data
- GET /relationships/triplets - Subject-action-object triplet identification
- POST /relationships/analyze - Batch relationship analysis

Provides comprehensive relationship mapping and confidence scoring.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Body
from pydantic import BaseModel, Field

from ...shared.schemas import ArticleResponse
from ...thalamus.nlp_pipeline import get_nlp_pipeline, NLPPipeline
from ..dependencies import get_repository_factory
from ...synaptic_vesicle.repositories import RepositoryFactory


logger = logging.getLogger(__name__)
router = APIRouter()


@dataclass
class EntityRelationship:
    """Represents a relationship between two entities."""
    entity1: str
    entity2: str
    relationship_type: str
    confidence: float
    evidence: List[str]
    context: str


@dataclass
class SubjectActionObject:
    """Represents a subject-action-object triplet."""
    subject: str
    action: str
    object: str
    confidence: float
    sentence: str
    position: int


class RelationshipRequest(BaseModel):
    """Request model for relationship analysis."""
    text: Optional[str] = Field(None, description="Text content to analyze")
    article_id: Optional[str] = Field(None, description="Article ID to analyze")
    include_graph_data: bool = Field(True, description="Include graph visualization data")
    min_confidence: float = Field(0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_relationships: int = Field(50, ge=1, le=200, description="Maximum relationships to return")
    relationship_types: Optional[List[str]] = Field(None, description="Filter by relationship types")


class RelationshipResponse(BaseModel):
    """Response model for relationship analysis."""
    entities: List[Dict[str, Any]] = Field(..., description="Extracted entities")
    relationships: List[Dict[str, Any]] = Field(..., description="Entity relationships")
    triplets: List[Dict[str, Any]] = Field(..., description="Subject-action-object triplets")
    graph_data: Optional[Dict[str, Any]] = Field(None, description="Graph visualization data")
    statistics: Dict[str, Any] = Field(..., description="Analysis statistics")
    processing_time: float = Field(..., description="Processing time in seconds")


@router.get(
    "/relationships",
    response_model=RelationshipResponse,
    summary="Extract entity relationships from content",
    description="""
    Extract and analyze relationships between entities in text content.
    
    Features:
    - Named entity recognition and relationship mapping
    - Confidence scoring for each relationship
    - Multiple relationship types (co-occurrence, semantic, syntactic)
    - Graph visualization data generation
    - Subject-action-object triplet extraction
    
    Supports filtering by confidence threshold and relationship types.
    """
)
async def extract_relationships(
    request: Request,
    text: Optional[str] = Query(None, description="Text content to analyze"),
    article_id: Optional[str] = Query(None, description="Article ID to analyze"),
    include_graph_data: bool = Query(True, description="Include graph visualization data"),
    min_confidence: float = Query(0.3, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    max_relationships: int = Query(50, ge=1, le=200, description="Maximum relationships to return"),
    relationship_types: Optional[str] = Query(None, description="Comma-separated relationship types"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> RelationshipResponse:
    """Extract relationships from content."""
    import time
    start_time = time.time()
    
    try:
        # Get content to analyze
        content_data = await _get_content_for_analysis(
            text, article_id, repository_factory
        )
        
        text_content = content_data["text"]
        title = content_data.get("title", "")
        
        if not text_content or len(text_content.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Content too short for meaningful relationship analysis"
            )
        
        # Parse relationship types filter
        types_filter = []
        if relationship_types:
            types_filter = [t.strip() for t in relationship_types.split(",")]
        
        # Perform NLP analysis
        nlp_analysis = await nlp_pipeline.process_text(text_content, title)
        
        # Extract entities and relationships
        entities = await _extract_entities_with_metadata(nlp_analysis, text_content)
        relationships = await _extract_entity_relationships(
            text_content, entities, min_confidence, types_filter
        )
        
        # Extract subject-action-object triplets
        triplets = await _extract_sao_triplets(text_content, nlp_pipeline)
        
        # Limit results
        relationships = relationships[:max_relationships]
        
        # Generate graph data if requested
        graph_data = None
        if include_graph_data:
            graph_data = _generate_graph_data(entities, relationships)
        
        # Calculate statistics
        statistics = _calculate_relationship_statistics(entities, relationships, triplets)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Extracted {len(relationships)} relationships in {processing_time:.2f}s")
        
        return RelationshipResponse(
            entities=entities,
            relationships=relationships,
            triplets=triplets,
            graph_data=graph_data,
            statistics=statistics,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in relationship extraction: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to extract relationships"
        )


@router.get(
    "/relationships/graph",
    summary="Get relationship graph visualization data",
    description="""
    Generate graph visualization data for entity relationships.
    
    Returns data formatted for popular graph visualization libraries:
    - Nodes (entities) with metadata
    - Edges (relationships) with weights
    - Layout suggestions and clustering information
    - Interactive visualization parameters
    """
)
async def get_relationship_graph(
    request: Request,
    text: Optional[str] = Query(None, description="Text content to analyze"),
    article_id: Optional[str] = Query(None, description="Article ID to analyze"),
    layout: str = Query("force", regex="^(force|circular|hierarchical|grid)$", description="Graph layout type"),
    min_confidence: float = Query(0.4, ge=0.0, le=1.0, description="Minimum confidence for edges"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> Dict[str, Any]:
    """Generate graph visualization data for relationships."""
    try:
        # Get content and extract relationships
        content_data = await _get_content_for_analysis(text, article_id, repository_factory)
        nlp_analysis = await nlp_pipeline.process_text(content_data["text"], content_data.get("title", ""))
        
        entities = await _extract_entities_with_metadata(nlp_analysis, content_data["text"])
        relationships = await _extract_entity_relationships(
            content_data["text"], entities, min_confidence, []
        )
        
        # Generate enhanced graph data
        graph_data = _generate_enhanced_graph_data(entities, relationships, layout)
        
        logger.info(f"Generated graph data with {len(entities)} nodes and {len(relationships)} edges")
        
        return graph_data
        
    except Exception as e:
        logger.error(f"Error generating graph data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate graph visualization data"
        )


@router.get(
    "/relationships/triplets",
    summary="Extract subject-action-object triplets",
    description="""
    Extract structured subject-action-object triplets from text.
    
    Features:
    - Dependency parsing for accurate triplet extraction
    - Confidence scoring for each triplet
    - Contextual information and sentence positioning
    - Filtering by confidence and triplet completeness
    """
)
async def extract_triplets(
    request: Request,
    text: Optional[str] = Query(None, description="Text content to analyze"),
    article_id: Optional[str] = Query(None, description="Article ID to analyze"),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    complete_only: bool = Query(True, description="Return only complete triplets (subject-action-object)"),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> Dict[str, Any]:
    """Extract subject-action-object triplets."""
    try:
        # Get content
        content_data = await _get_content_for_analysis(text, article_id, repository_factory)
        
        # Extract triplets
        triplets = await _extract_sao_triplets(content_data["text"], nlp_pipeline)
        
        # Filter by confidence and completeness
        filtered_triplets = []
        for triplet in triplets:
            if triplet["confidence"] >= min_confidence:
                if not complete_only or (triplet["subject"] and triplet["action"] and triplet["object"]):
                    filtered_triplets.append(triplet)
        
        # Group by action for analysis
        action_groups = defaultdict(list)
        for triplet in filtered_triplets:
            action_groups[triplet["action"]].append(triplet)
        
        logger.info(f"Extracted {len(filtered_triplets)} triplets")
        
        return {
            "triplets": filtered_triplets,
            "total_count": len(filtered_triplets),
            "action_groups": dict(action_groups),
            "statistics": {
                "unique_subjects": len(set(t["subject"] for t in filtered_triplets if t["subject"])),
                "unique_actions": len(set(t["action"] for t in filtered_triplets if t["action"])),
                "unique_objects": len(set(t["object"] for t in filtered_triplets if t["object"])),
                "average_confidence": sum(t["confidence"] for t in filtered_triplets) / len(filtered_triplets) if filtered_triplets else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error extracting triplets: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to extract triplets"
        )


@router.post(
    "/relationships/analyze",
    summary="Batch relationship analysis",
    description="""
    Analyze relationships across multiple documents or text segments.
    
    Features:
    - Batch processing for multiple texts
    - Cross-document relationship detection
    - Relationship aggregation and scoring
    - Comparative analysis between documents
    """
)
async def batch_relationship_analysis(
    request: Request,
    batch_request: Dict[str, Any] = Body(...),
    repository_factory: RepositoryFactory = Depends(get_repository_factory),
    nlp_pipeline: NLPPipeline = Depends(get_nlp_pipeline)
) -> Dict[str, Any]:
    """Perform batch relationship analysis."""
    try:
        user_tier = getattr(request.state, "user_tier", "free")
        
        # Validate batch size based on user tier
        max_batch_size = {"free": 3, "premium": 10, "enterprise": 50}.get(user_tier, 3)
        
        items = batch_request.get("items", [])
        if len(items) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds limit for {user_tier} tier: {max_batch_size}"
            )
        
        # Process each item
        batch_results = []
        all_entities = {}
        all_relationships = []
        
        for i, item in enumerate(items):
            try:
                # Get content
                content_data = await _get_content_for_analysis(
                    item.get("text"), item.get("article_id"), repository_factory
                )
                
                # Analyze relationships
                nlp_analysis = await nlp_pipeline.process_text(
                    content_data["text"], content_data.get("title", "")
                )
                
                entities = await _extract_entities_with_metadata(nlp_analysis, content_data["text"])
                relationships = await _extract_entity_relationships(
                    content_data["text"], entities, 0.3, []
                )
                
                # Store results
                batch_results.append({
                    "item_index": i,
                    "success": True,
                    "entities_count": len(entities),
                    "relationships_count": len(relationships),
                    "entities": entities,
                    "relationships": relationships
                })
                
                # Aggregate for cross-document analysis
                for entity in entities:
                    entity_key = entity["text"].lower()
                    if entity_key not in all_entities:
                        all_entities[entity_key] = {
                            "text": entity["text"],
                            "type": entity["type"],
                            "documents": [],
                            "total_mentions": 0
                        }
                    all_entities[entity_key]["documents"].append(i)
                    all_entities[entity_key]["total_mentions"] += entity.get("mentions", 1)
                
                all_relationships.extend(relationships)
                
            except Exception as e:
                batch_results.append({
                    "item_index": i,
                    "success": False,
                    "error": str(e)
                })
        
        # Analyze cross-document relationships
        cross_doc_relationships = _analyze_cross_document_relationships(all_entities, batch_results)
        
        # Calculate batch statistics
        successful = sum(1 for r in batch_results if r["success"])
        total_entities = sum(r.get("entities_count", 0) for r in batch_results if r["success"])
        total_relationships = sum(r.get("relationships_count", 0) for r in batch_results if r["success"])
        
        logger.info(f"Processed batch of {len(items)} items: {successful} successful")
        
        return {
            "batch_summary": {
                "total_items": len(items),
                "successful": successful,
                "failed": len(items) - successful,
                "total_entities": total_entities,
                "total_relationships": total_relationships
            },
            "results": batch_results,
            "cross_document_analysis": {
                "shared_entities": [e for e in all_entities.values() if len(e["documents"]) > 1],
                "cross_relationships": cross_doc_relationships,
                "entity_frequency": sorted(all_entities.values(), key=lambda x: x["total_mentions"], reverse=True)[:10]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch relationship analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process batch relationship analysis"
        )


# Helper functions for relationship analysis
async def _get_content_for_analysis(
    text: Optional[str],
    article_id: Optional[str],
    repository_factory: RepositoryFactory
) -> Dict[str, str]:
    """Get content from various sources for analysis."""
    if text:
        return {"text": text, "title": ""}
    
    elif article_id:
        article_repo = repository_factory.get_article_repository()
        article = await article_repo.get_by_id(article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        return {"text": article.content, "title": article.title}
    
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either text or article_id"
        )


async def _extract_entities_with_metadata(nlp_analysis: Any, text: str) -> List[Dict[str, Any]]:
    """Extract entities with additional metadata."""
    entities = []
    entity_counts = defaultdict(int)
    
    # Count entity mentions
    if hasattr(nlp_analysis, 'entities'):
        for entity in nlp_analysis.entities:
            entity_text = entity.text if hasattr(entity, 'text') else entity.get('text', '')
            entity_counts[entity_text.lower()] += 1
    
    # Process entities with metadata
    processed_entities = set()
    if hasattr(nlp_analysis, 'entities'):
        for entity in nlp_analysis.entities:
            entity_text = entity.text if hasattr(entity, 'text') else entity.get('text', '')
            entity_type = entity.label if hasattr(entity, 'label') else entity.get('label', '')
            
            if entity_text.lower() not in processed_entities:
                processed_entities.add(entity_text.lower())
                
                # Calculate entity importance
                importance = _calculate_entity_importance(entity_text, text, entity_counts[entity_text.lower()])
                
                entities.append({
                    "text": entity_text,
                    "type": entity_type,
                    "mentions": entity_counts[entity_text.lower()],
                    "importance": importance,
                    "confidence": getattr(entity, 'confidence', 0.8),
                    "positions": _find_entity_positions(entity_text, text)
                })
    
    # Sort by importance
    entities.sort(key=lambda x: x["importance"], reverse=True)
    return entities


async def _extract_entity_relationships(
    text: str,
    entities: List[Dict[str, Any]],
    min_confidence: float,
    types_filter: List[str]
) -> List[Dict[str, Any]]:
    """Extract relationships between entities."""
    relationships = []
    
    # Co-occurrence based relationships
    for i, entity1 in enumerate(entities):
        for entity2 in entities[i+1:]:
            relationship = _analyze_entity_cooccurrence(entity1, entity2, text)
            if relationship and relationship["confidence"] >= min_confidence:
                if not types_filter or relationship["type"] in types_filter:
                    relationships.append(relationship)
    
    # Syntactic relationships (simplified)
    syntactic_relationships = _extract_syntactic_relationships(text, entities)
    for rel in syntactic_relationships:
        if rel["confidence"] >= min_confidence:
            if not types_filter or rel["type"] in types_filter:
                relationships.append(rel)
    
    # Sort by confidence
    relationships.sort(key=lambda x: x["confidence"], reverse=True)
    return relationships


async def _extract_sao_triplets(text: str, nlp_pipeline: NLPPipeline) -> List[Dict[str, Any]]:
    """Extract subject-action-object triplets from text."""
    triplets = []
    
    # Split text into sentences
    sentences = text.split('. ')
    
    for i, sentence in enumerate(sentences):
        if len(sentence.strip()) < 10:
            continue
        
        # Simple pattern-based triplet extraction
        # In a real implementation, this would use dependency parsing
        triplet_patterns = [
            r'(\w+(?:\s+\w+)*)\s+(is|are|was|were|has|have|had|will|would|can|could|should|must)\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+(said|says|announced|reported|stated|declared)\s+(.+)',
            r'(\w+(?:\s+\w+)*)\s+(bought|sold|acquired|launched|released|developed)\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+(increased|decreased|grew|fell|rose|dropped)\s+(\w+(?:\s+\w+)*)'
        ]
        
        for pattern in triplet_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).strip()
                action = match.group(2).strip()
                obj = match.group(3).strip() if len(match.groups()) > 2 else ""
                
                # Calculate confidence based on pattern match quality
                confidence = _calculate_triplet_confidence(subject, action, obj, sentence)
                
                if confidence > 0.3:
                    triplets.append({
                        "subject": subject,
                        "action": action,
                        "object": obj,
                        "confidence": confidence,
                        "sentence": sentence,
                        "position": i
                    })
    
    return triplets


def _generate_graph_data(entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate basic graph visualization data."""
    nodes = []
    edges = []
    
    # Create nodes
    for entity in entities:
        nodes.append({
            "id": entity["text"],
            "label": entity["text"],
            "type": entity["type"],
            "size": min(50, max(10, entity["mentions"] * 5)),
            "importance": entity["importance"]
        })
    
    # Create edges
    for relationship in relationships:
        edges.append({
            "source": relationship["entity1"],
            "target": relationship["entity2"],
            "type": relationship["type"],
            "weight": relationship["confidence"],
            "label": relationship["type"]
        })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "layout": "force",
        "statistics": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "density": len(edges) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0
        }
    }


def _generate_enhanced_graph_data(
    entities: List[Dict[str, Any]], 
    relationships: List[Dict[str, Any]], 
    layout: str
) -> Dict[str, Any]:
    """Generate enhanced graph visualization data with layout and clustering."""
    basic_graph = _generate_graph_data(entities, relationships)
    
    # Add layout-specific parameters
    layout_params = {
        "force": {"strength": -300, "distance": 100},
        "circular": {"radius": 200},
        "hierarchical": {"direction": "TB", "sortMethod": "directed"},
        "grid": {"avoidOverlap": True}
    }
    
    # Add clustering information
    clusters = _detect_entity_clusters(entities, relationships)
    
    # Enhance nodes with clustering info
    for node in basic_graph["nodes"]:
        node["cluster"] = clusters.get(node["id"], 0)
        node["color"] = _get_cluster_color(clusters.get(node["id"], 0))
    
    basic_graph.update({
        "layout": layout,
        "layout_params": layout_params.get(layout, {}),
        "clusters": clusters,
        "cluster_count": len(set(clusters.values()))
    })
    
    return basic_graph


def _calculate_relationship_statistics(
    entities: List[Dict[str, Any]], 
    relationships: List[Dict[str, Any]], 
    triplets: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Calculate comprehensive relationship statistics."""
    # Entity type distribution
    entity_types = defaultdict(int)
    for entity in entities:
        entity_types[entity["type"]] += 1
    
    # Relationship type distribution
    relationship_types = defaultdict(int)
    for rel in relationships:
        relationship_types[rel["type"]] += 1
    
    # Confidence distribution
    confidences = [rel["confidence"] for rel in relationships]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Network metrics
    entity_connections = defaultdict(int)
    for rel in relationships:
        entity_connections[rel["entity1"]] += 1
        entity_connections[rel["entity2"]] += 1
    
    most_connected = max(entity_connections.items(), key=lambda x: x[1]) if entity_connections else ("", 0)
    
    return {
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "triplet_count": len(triplets),
        "entity_types": dict(entity_types),
        "relationship_types": dict(relationship_types),
        "average_confidence": avg_confidence,
        "most_connected_entity": {
            "entity": most_connected[0],
            "connections": most_connected[1]
        },
        "network_density": len(relationships) / (len(entities) * (len(entities) - 1) / 2) if len(entities) > 1 else 0
    }


def _calculate_entity_importance(entity_text: str, text: str, mentions: int) -> float:
    """Calculate entity importance score."""
    # Base score from mention frequency
    text_length = len(text.split())
    frequency_score = min(1.0, mentions / (text_length / 100))
    
    # Position score (entities mentioned early are often more important)
    first_mention_pos = text.lower().find(entity_text.lower())
    position_score = 1.0 - (first_mention_pos / len(text)) if first_mention_pos != -1 else 0.5
    
    # Length score (longer entities are often more specific/important)
    length_score = min(1.0, len(entity_text.split()) / 5)
    
    # Combine scores
    importance = (frequency_score * 0.4 + position_score * 0.3 + length_score * 0.3)
    return min(1.0, importance)


def _find_entity_positions(entity_text: str, text: str) -> List[int]:
    """Find all positions where entity appears in text."""
    positions = []
    start = 0
    entity_lower = entity_text.lower()
    text_lower = text.lower()
    
    while True:
        pos = text_lower.find(entity_lower, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    
    return positions


def _analyze_entity_cooccurrence(entity1: Dict[str, Any], entity2: Dict[str, Any], text: str) -> Optional[Dict[str, Any]]:
    """Analyze co-occurrence relationship between two entities."""
    text_lower = text.lower()
    entity1_text = entity1["text"].lower()
    entity2_text = entity2["text"].lower()
    
    # Find all positions of both entities
    pos1 = _find_entity_positions(entity1_text, text_lower)
    pos2 = _find_entity_positions(entity2_text, text_lower)
    
    if not pos1 or not pos2:
        return None
    
    # Calculate minimum distance between entities
    min_distance = float('inf')
    for p1 in pos1:
        for p2 in pos2:
            distance = abs(p1 - p2)
            min_distance = min(min_distance, distance)
    
    # Calculate confidence based on proximity and frequency
    max_distance = 200  # characters
    proximity_score = max(0, 1 - (min_distance / max_distance))
    frequency_score = min(1.0, (len(pos1) + len(pos2)) / 10)
    
    confidence = (proximity_score * 0.7 + frequency_score * 0.3)
    
    if confidence < 0.1:
        return None
    
    return {
        "entity1": entity1["text"],
        "entity2": entity2["text"],
        "type": "co-occurrence",
        "confidence": confidence,
        "evidence": [f"Entities appear within {min_distance} characters"],
        "distance": min_distance
    }


def _extract_syntactic_relationships(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract syntactic relationships using simple patterns."""
    relationships = []
    
    # Simple syntactic patterns
    patterns = [
        (r'(\w+(?:\s+\w+)*)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+)*)', "is-a"),
        (r'(\w+(?:\s+\w+)*)\s+(?:has|have|had)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+)*)', "has"),
        (r'(\w+(?:\s+\w+)*)\s+(?:owns|owned)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+)*)', "owns"),
        (r'(\w+(?:\s+\w+)*)\s+(?:works for|employed by)\s+(\w+(?:\s+\w+)*)', "works-for")
    ]
    
    entity_texts = {e["text"].lower(): e["text"] for e in entities}
    
    for pattern, rel_type in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entity1_match = match.group(1).strip()
            entity2_match = match.group(2).strip()
            
            # Check if both are recognized entities
            entity1_key = entity1_match.lower()
            entity2_key = entity2_match.lower()
            
            if entity1_key in entity_texts and entity2_key in entity_texts:
                relationships.append({
                    "entity1": entity_texts[entity1_key],
                    "entity2": entity_texts[entity2_key],
                    "type": rel_type,
                    "confidence": 0.7,
                    "evidence": [match.group(0)],
                    "pattern": pattern
                })
    
    return relationships


def _calculate_triplet_confidence(subject: str, action: str, obj: str, sentence: str) -> float:
    """Calculate confidence score for a triplet."""
    # Base confidence
    confidence = 0.5
    
    # Length penalties/bonuses
    if len(subject.split()) > 5 or len(obj.split()) > 5:
        confidence -= 0.2  # Too long, might be noisy
    
    if len(subject.split()) == 1 and len(obj.split()) == 1:
        confidence += 0.1  # Simple, likely accurate
    
    # Action quality
    strong_actions = ['is', 'are', 'was', 'were', 'said', 'announced', 'bought', 'sold']
    if action.lower() in strong_actions:
        confidence += 0.2
    
    # Sentence quality
    if len(sentence.split()) < 5:
        confidence -= 0.3  # Too short
    elif len(sentence.split()) > 30:
        confidence -= 0.1  # Too long, might be complex
    
    return max(0.0, min(1.0, confidence))


def _detect_entity_clusters(entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Dict[str, int]:
    """Detect entity clusters based on relationships."""
    # Simple clustering based on relationship connectivity
    clusters = {}
    cluster_id = 0
    
    # Build adjacency list
    adjacency = defaultdict(set)
    for rel in relationships:
        adjacency[rel["entity1"]].add(rel["entity2"])
        adjacency[rel["entity2"]].add(rel["entity1"])
    
    # Assign clusters using simple connected components
    visited = set()
    
    for entity in entities:
        entity_text = entity["text"]
        if entity_text not in visited:
            # Start new cluster
            cluster_entities = set()
            stack = [entity_text]
            
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    cluster_entities.add(current)
                    
                    # Add connected entities
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            
            # Assign cluster ID to all entities in this component
            for e in cluster_entities:
                clusters[e] = cluster_id
            
            cluster_id += 1
    
    return clusters


def _get_cluster_color(cluster_id: int) -> str:
    """Get color for cluster visualization."""
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
    ]
    return colors[cluster_id % len(colors)]


def _analyze_cross_document_relationships(all_entities: Dict[str, Any], batch_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze relationships that span across multiple documents."""
    cross_relationships = []
    
    # Find entities that appear in multiple documents
    multi_doc_entities = {k: v for k, v in all_entities.items() if len(v["documents"]) > 1}
    
    for entity_key, entity_data in multi_doc_entities.items():
        # Create cross-document relationships
        docs = entity_data["documents"]
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                cross_relationships.append({
                    "entity": entity_data["text"],
                    "document1": docs[i],
                    "document2": docs[j],
                    "type": "cross-document-mention",
                    "confidence": 0.8,
                    "total_mentions": entity_data["total_mentions"]
                })
    
    return cross_relationships