"""
Data loader utility for Enhanced SentimentR
"""
import os
from pathlib import Path
from typing import List, Set, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Utility class for loading lexicon and data files"""
    
    def __init__(self):
        """Initialize data loader with lexica directory"""
        # Find the lexica directory relative to this file
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        
        # Try different possible locations for lexica
        possible_paths = [
            project_root / "sentimentr" / "lexica",  # New structure
            project_root / "lexica",  # Root level
            current_dir / "lexica",  # Utils level
        ]
        
        self.lexica_dir = None
        for path in possible_paths:
            if path.exists():
                self.lexica_dir = path
                break
        
        if not self.lexica_dir:
            # Use the original sentimentr lexica as fallback
            original_sentimentr_path = project_root.parent / "sentimentr" / "lexica"
            if original_sentimentr_path.exists():
                self.lexica_dir = original_sentimentr_path
            else:
                logger.warning("No lexica directory found")
                self.lexica_dir = project_root  # Fallback to project root
    
    def load_word_list(self, filename: str) -> Set[str]:
        """
        Load a word list from a lexicon file
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Set of words from the file
        """
        if not self.lexica_dir:
            logger.warning("No lexica directory available")
            return set()
            
        file_path = self.lexica_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Lexicon file not found: {file_path}")
            return set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                words = set()
                for line in f:
                    word = line.strip().lower()
                    if word and not word.startswith('#'):  # Skip comments
                        words.add(word)
                return words
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return set()
    
    def load_word_scores(self, filename: str) -> Dict[str, float]:
        """
        Load word scores from a lexicon file (format: word score)
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Dictionary mapping words to scores
        """
        if not self.lexica_dir:
            logger.warning("No lexica directory available")
            return {}
            
        file_path = self.lexica_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Lexicon file not found: {file_path}")
            return {}
        
        try:
            word_scores = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            word = parts[0].lower()
                            try:
                                score = float(parts[1])
                                word_scores[word] = score
                            except ValueError:
                                logger.warning(f"Invalid score in {filename}: {line}")
            return word_scores
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return {}
    
    def list_available_files(self) -> List[str]:
        """List all available lexicon files"""
        if not self.lexica_dir or not self.lexica_dir.exists():
            return []
        
        return [f.name for f in self.lexica_dir.iterdir() if f.is_file()]
    
    def get_lexica_path(self) -> Optional[Path]:
        """Get the path to the lexica directory"""
        return self.lexica_dir


# Global instance for easy access
data_loader = DataLoader()
