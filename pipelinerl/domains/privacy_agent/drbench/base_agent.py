from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, TypedDict

class InsightDict(TypedDict):
    """Structure for individual insights with claims and citations."""
    claim: str
    citations: List[str]


class BaseAgent(ABC):
    """
    Abstract base class for all Deep Research agents.
    This class defines the common interface that all agents must implement.
    """
    
    @abstractmethod
    def generate_report(self, query: str, extract_insights: bool = False, **kwargs) -> Tuple[str, Optional[List[InsightDict]]]:
        """
        Generate a research report based on the query.
        
        Args:
            query (str): The research question
            extract_insights (bool): If True, extract and return insights with citations from report; if False, return None
            **kwargs: Additional arguments specific to the agent implementation
            
        Returns:
            Tuple[str, Optional[List[InsightDict]]]: (report_text, insights) where insights is a list of claim-citation pairs or None
        """
        pass
    
    @abstractmethod
    def get_report_metadata(self):
        """
        Get metadata about the generated report.
        
        Returns:
            dict: Metadata about the report, including document references
        """
        pass
    
    @abstractmethod
    def save_report(self, save_path, **kwargs):
        """
        Save the report to a file.
        
        Args:
            save_path (str): Path to save the report
            **kwargs: Additional arguments specific to the agent implementation
            
        Returns:
            dict: Result of the save operation
        """
        pass 