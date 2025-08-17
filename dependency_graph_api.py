"""
FastAPI Web Interface for GitHub Repository Dependency Graph Builder

This module provides a web API for the dependency graph builder, allowing users to:
- Analyze GitHub repositories through REST endpoints
- Query dependency relationships
- Monitor repository changes
- Visualize dependency graphs
- Detect circular dependencies and impact analysis

The API follows FastAPI best practices and integrates with the core dependency_graph_builder module.

Endpoints:
- POST /analyze-repository - Analyze a GitHub repository
- GET /repository/{repo_name}/stats - Get repository statistics  
- GET /repository/{repo_name}/dependencies/{file_path} - Get file dependencies
- GET /repository/{repo_name}/cycles - Find circular dependencies
- POST /repository/{repo_name}/update - Update repository analysis
- GET /health - API health check
"""

import os
import asyncio
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Import our dependency graph builder
from dependency_graph_builder import DependencyGraphBuilder

# Models for API requests/responses
class RepositoryAnalysisRequest(BaseModel):
    """Request model for repository analysis"""
    repo_name: str = Field(..., description="GitHub repository name in format 'owner/repo'")
    clear_existing: bool = Field(True, description="Whether to clear existing data before analysis")
    include_tests: bool = Field(False, description="Whether to include test files in analysis")

class RepositoryAnalysisResponse(BaseModel):
    """Response model for repository analysis"""
    repository: str
    status: str
    files_processed: int
    total_imports: int
    file_types: List[str]
    statistics: Dict[str, Any]
    analysis_time: Optional[str] = None
    message: Optional[str] = None

class FileLocationRequest(BaseModel):
    """Request model for file location queries"""
    repo_name: str
    file_path: str

class CircularDependencyResponse(BaseModel):
    """Response model for circular dependency detection"""
    repository: str
    cycles_found: int
    cycles: List[List[str]]
    severity: str  # 'none', 'low', 'medium', 'high'

class DependencyQueryResponse(BaseModel):
    """Response model for dependency queries"""
    file: str
    repository: str
    direct_dependencies: List[Dict[str, Any]]
    reverse_dependencies: List[Dict[str, Any]]
    dependency_depth: int

class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    neo4j_connected: bool
    github_token_configured: bool
    active_repositories: int
    total_files_tracked: int
    last_update: Optional[str] = None

# Create FastAPI instance
app = FastAPI(
    title="GitHub Dependency Graph Builder API",
    description="Analyze GitHub repositories and build dependency graphs using Neo4j",
    version="1.0.0"
)

# Global dependency graph builder instance
builder: Optional[DependencyGraphBuilder] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the dependency graph builder on startup"""
    global builder
    try:
        builder = DependencyGraphBuilder()
        print("✅ Dependency Graph Builder API started successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Dependency Graph Builder: {e}")
        # We'll still start the API but endpoints will return errors

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global builder
    if builder:
        builder.close()
        print("✅ Dependency Graph Builder API shutdown complete")

def get_builder() -> DependencyGraphBuilder:
    """Get the dependency graph builder instance with error handling"""
    if builder is None:
        raise HTTPException(
            status_code=500, 
            detail="Dependency Graph Builder not initialized. Check Neo4j and GitHub configuration."
        )
    return builder

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Serve a simple web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GitHub Dependency Graph Builder</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #2196F3; }
            .description { color: #666; margin-top: 5px; }
            h1 { color: #333; }
            code { background: #e8e8e8; padding: 2px 4px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔗 GitHub Dependency Graph Builder API</h1>
            <p>Build and analyze dependency graphs from GitHub repositories using Neo4j.</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <div class="method">POST /analyze-repository</div>
                <div class="description">Analyze a GitHub repository and build its dependency graph</div>
            </div>
            
            <div class="endpoint">
                <div class="method">GET /repository/{repo_name}/stats</div>
                <div class="description">Get statistics for a repository's dependency graph</div>
            </div>
            
            <div class="endpoint">
                <div class="method">GET /repository/{repo_name}/dependencies/{file_path}</div>
                <div class="description">Get dependencies for a specific file</div>
            </div>
            
            <div class="endpoint">
                <div class="method">GET /repository/{repo_name}/cycles</div>
                <div class="description">Find circular dependencies in a repository</div>
            </div>
            
            <div class="endpoint">
                <div class="method">GET /health</div>
                <div class="description">Check API health and system status</div>
            </div>
            
            <h2>Quick Start:</h2>
            <p>1. Check system health: <code>GET /health</code></p>
            <p>2. Analyze a repository: <code>POST /analyze-repository</code> with <code>{"repo_name": "owner/repo"}</code></p>
            <p>3. View repository stats: <code>GET /repository/owner%2Frepo/stats</code></p>
            
            <h2>Documentation:</h2>
            <p>Visit <a href="/docs">/docs</a> for interactive API documentation.</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/analyze-repository", response_model=RepositoryAnalysisResponse)
async def analyze_repository(
    request: RepositoryAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze a GitHub repository and build its dependency graph
    
    This endpoint scans the specified repository, parses all supported file types,
    and builds a comprehensive dependency graph stored in Neo4j.
    """
    try:
        builder_instance = get_builder()
        start_time = datetime.now()
        
        # Validate repository name format
        if '/' not in request.repo_name:
            raise HTTPException(
                status_code=400,
                detail="Repository name must be in format 'owner/repo'"
            )
        
        print(f"🚀 Starting analysis of repository: {request.repo_name}")
        
        # Build the dependency graph
        result = builder_instance.build_repository_graph(
            repo_name=request.repo_name,
            clear_existing=request.clear_existing
        )
        
        analysis_time = str(datetime.now() - start_time)
        
        if result.get("status") == "success":
            return RepositoryAnalysisResponse(
                repository=request.repo_name,
                status="success",
                files_processed=result["files_processed"],
                total_imports=result["total_imports"],
                file_types=result["file_types"],
                statistics=result["statistics"],
                analysis_time=analysis_time,
                message=f"Successfully analyzed {result['files_processed']} files"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {result.get('error', 'Unknown error')}"
            )
    
    except Exception as e:
        print(f"❌ Error analyzing repository {request.repo_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repository/{repo_owner}/{repo_name}/stats")
async def get_repository_stats(repo_owner: str, repo_name: str):
    """Get statistics for a repository's dependency graph"""
    try:
        builder_instance = get_builder()
        full_repo_name = f"{repo_owner}/{repo_name}"
        
        stats = builder_instance.get_repository_statistics(full_repo_name)
        
        if "error" in stats:
            raise HTTPException(status_code=404, detail=f"Repository {full_repo_name} not found or error occurred")
        
        return {
            "repository": full_repo_name,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"❌ Error getting repository stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repository/{repo_owner}/{repo_name}/dependencies")
async def get_file_dependencies(
    repo_owner: str, 
    repo_name: str, 
    file_path: str
):
    """Get dependencies for a specific file in a repository"""
    try:
        builder_instance = get_builder()
        full_repo_name = f"{repo_owner}/{repo_name}"
        
        dependencies = builder_instance.get_file_dependencies(full_repo_name, file_path)
        
        if "error" in dependencies:
            raise HTTPException(status_code=404, detail=f"File {file_path} not found or error occurred")
        
        return DependencyQueryResponse(
            file=file_path,
            repository=full_repo_name,
            direct_dependencies=dependencies["direct_dependencies"],
            reverse_dependencies=dependencies["reverse_dependencies"],
            dependency_depth=len(dependencies["direct_dependencies"])
        )
    
    except Exception as e:
        print(f"❌ Error getting file dependencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repository/{repo_owner}/{repo_name}/cycles", response_model=CircularDependencyResponse)
async def find_circular_dependencies(repo_owner: str, repo_name: str):
    """Find circular dependencies in a repository"""
    try:
        builder_instance = get_builder()
        full_repo_name = f"{repo_owner}/{repo_name}"
        
        cycles = builder_instance.find_circular_dependencies(full_repo_name)
        
        # Determine severity based on number and length of cycles
        severity = "none"
        if cycles:
            max_cycle_length = max(len(cycle) for cycle in cycles)
            if len(cycles) > 10 or max_cycle_length > 5:
                severity = "high"
            elif len(cycles) > 5 or max_cycle_length > 3:
                severity = "medium"
            else:
                severity = "low"
        
        return CircularDependencyResponse(
            repository=full_repo_name,
            cycles_found=len(cycles),
            cycles=cycles,
            severity=severity
        )
    
    except Exception as e:
        print(f"❌ Error finding circular dependencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/repository/{repo_owner}/{repo_name}/update")
async def update_repository_analysis(
    repo_owner: str, 
    repo_name: str,
    background_tasks: BackgroundTasks
):
    """Update the dependency analysis for a repository (incremental update)"""
    try:
        builder_instance = get_builder()
        full_repo_name = f"{repo_owner}/{repo_name}"
        
        # For now, we'll do a full re-analysis
        # In a production system, this would be an incremental update
        result = builder_instance.build_repository_graph(
            repo_name=full_repo_name,
            clear_existing=True
        )
        
        return {
            "repository": full_repo_name,
            "status": "updated" if result.get("status") == "success" else "failed",
            "message": "Repository analysis updated successfully" if result.get("status") == "success" else f"Update failed: {result.get('error')}",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"❌ Error updating repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repositories")
async def list_analyzed_repositories():
    """List all repositories that have been analyzed"""
    try:
        builder_instance = get_builder()
        
        # Query Neo4j for distinct repository names
        with builder_instance.driver.session() as session:
            result = session.run("""
                MATCH (f:File)
                RETURN DISTINCT f.repo_name as repo_name, 
                       count(f) as file_count,
                       max(f.last_modified) as last_updated
                ORDER BY last_updated DESC
            """)
            
            repositories = []
            for record in result:
                repositories.append({
                    "repository": record["repo_name"],
                    "file_count": record["file_count"],
                    "last_updated": record["last_updated"]
                })
            
            return {
                "repositories": repositories,
                "total_count": len(repositories),
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        print(f"❌ Error listing repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/dependencies")
async def search_dependencies(
    query: str,
    repo_name: Optional[str] = None,
    dependency_type: Optional[str] = None,
    limit: int = 10
):
    """Search for dependencies across repositories"""
    try:
        builder_instance = get_builder()
        
        # Build Cypher query based on parameters
        cypher_parts = ["MATCH (source:File)-[d:DEPENDS_ON]->(target)"]
        params = {"limit": limit}
        
        where_conditions = []
        
        if repo_name:
            where_conditions.append("source.repo_name = $repo_name")
            params["repo_name"] = repo_name
        
        if dependency_type:
            where_conditions.append("d.dependency_type = $dependency_type") 
            params["dependency_type"] = dependency_type
        
        # Search in file paths, imported names, or module names
        where_conditions.append("""
            (source.path CONTAINS $query OR 
             target.path CONTAINS $query OR 
             target.name CONTAINS $query OR 
             ANY(name IN d.imported_names WHERE name CONTAINS $query))
        """)
        params["query"] = query
        
        if where_conditions:
            cypher_parts.append("WHERE " + " AND ".join(where_conditions))
        
        cypher_parts.extend([
            "RETURN source.repo_name as repo, source.path as source_file,",
            "       target.path as target_file, target.name as target_name,",
            "       d.dependency_type as type, d.imported_names as imports",
            "LIMIT $limit"
        ])
        
        cypher_query = " ".join(cypher_parts)
        
        with builder_instance.driver.session() as session:
            result = session.run(cypher_query, params)
            
            dependencies = []
            for record in result:
                dependencies.append({
                    "repository": record["repo"],
                    "source_file": record["source_file"],
                    "target_file": record["target_file"],
                    "target_name": record["target_name"],
                    "dependency_type": record["type"],
                    "imported_names": record["imports"]
                })
            
            return {
                "query": query,
                "results": dependencies,
                "count": len(dependencies),
                "parameters": {
                    "repo_name": repo_name,
                    "dependency_type": dependency_type,
                    "limit": limit
                }
            }
    
    except Exception as e:
        print(f"❌ Error searching dependencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check the health of the API and its dependencies"""
    try:
        neo4j_connected = False
        github_configured = False
        active_repositories = 0
        total_files = 0
        
        # Check Neo4j connection
        try:
            if builder:
                with builder.driver.session() as session:
                    session.run("RETURN 1")
                neo4j_connected = True
                
                # Get counts
                result = session.run("MATCH (f:File) RETURN count(DISTINCT f.repo_name) as repos, count(f) as files")
                record = result.single()
                if record:
                    active_repositories = record["repos"]
                    total_files = record["files"]
            
        except Exception as e:
            print(f"Neo4j health check failed: {e}")
        
        # Check GitHub token
        try:
            github_token = os.getenv('GITHUB_PAT')
            github_configured = bool(github_token)
        except Exception as e:
            print(f"GitHub token check failed: {e}")
        
        # Overall status
        status = "healthy" if neo4j_connected and github_configured else "degraded"
        
        return HealthCheckResponse(
            status=status,
            neo4j_connected=neo4j_connected,
            github_token_configured=github_configured,
            active_repositories=active_repositories,
            total_files_tracked=total_files,
            last_update=datetime.now().isoformat()
        )
    
    except Exception as e:
        print(f"❌ Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "detail": "The requested resource was not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
