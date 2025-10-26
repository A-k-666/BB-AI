"""
Utility functions for Browserbase + n8n automation backend.
Handles click, type, connect nodes, logging and error handling.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class AutomationUtils:
    """Utility class for browser automation operations."""
    
    def __init__(self, driver: webdriver.Chrome, wait_timeout: int = 10):
        """
        Initialize automation utilities.
        
        Args:
            driver: Selenium WebDriver instance
            wait_timeout: Maximum wait time for elements (seconds)
        """
        self.driver = driver
        self.wait = WebDriverWait(driver, wait_timeout)
        self.actions = ActionChains(driver)
        self.logger = logging.getLogger(__name__)
    
    def click(self, selector: str, by: By = By.CSS_SELECTOR) -> bool:
        """
        Simulate click on element.
        
        Args:
            selector: CSS selector or XPath
            by: Locator strategy (default: CSS_SELECTOR)
            
        Returns:
            bool: True if click successful, False otherwise
        """
        try:
            element = self.wait.until(EC.element_to_be_clickable((by, selector)))
            element.click()
            self.logger.info(f"Successfully clicked element: {selector}")
            return True
        except (TimeoutException, NoSuchElementException) as e:
            self.logger.error(f"Failed to click element {selector}: {e}")
            return False
    
    def type_text(self, selector: str, text: str, by: By = By.CSS_SELECTOR, clear_first: bool = True) -> bool:
        """
        Type text into input field.
        
        Args:
            selector: CSS selector or XPath
            text: Text to type
            by: Locator strategy
            clear_first: Whether to clear field before typing
            
        Returns:
            bool: True if typing successful, False otherwise
        """
        try:
            element = self.wait.until(EC.presence_of_element_located((by, selector)))
            
            if clear_first:
                element.clear()
            
            element.send_keys(text)
            self.logger.info(f"Successfully typed text into: {selector}")
            return True
        except (TimeoutException, NoSuchElementException) as e:
            self.logger.error(f"Failed to type text into {selector}: {e}")
            return False
    
    def connect_nodes(self, from_node_id: str, to_node_id: str) -> bool:
        """
        Connect two n8n workflow nodes.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Find source node
            source_selector = f"[data-node-id='{from_node_id}'] .node-output"
            source_element = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, source_selector)))
            
            # Find target node
            target_selector = f"[data-node-id='{to_node_id}'] .node-input"
            target_element = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, target_selector)))
            
            # Drag from source to target
            self.actions.drag_and_drop(source_element, target_element).perform()
            
            self.logger.info(f"Successfully connected nodes: {from_node_id} -> {to_node_id}")
            return True
        except (TimeoutException, NoSuchElementException) as e:
            self.logger.error(f"Failed to connect nodes {from_node_id} -> {to_node_id}: {e}")
            return False
    
    def wait_for_element(self, selector: str, by: By = By.CSS_SELECTOR, timeout: int = 10) -> bool:
        """
        Wait for element to be present and visible.
        
        Args:
            selector: CSS selector or XPath
            by: Locator strategy
            timeout: Maximum wait time
            
        Returns:
            bool: True if element found, False otherwise
        """
        try:
            self.wait.until(EC.presence_of_element_located((by, selector)))
            self.logger.info(f"Element found: {selector}")
            return True
        except TimeoutException:
            self.logger.warning(f"Element not found within timeout: {selector}")
            return False
    
    def get_element_text(self, selector: str, by: By = By.CSS_SELECTOR) -> Optional[str]:
        """
        Get text content of element.
        
        Args:
            selector: CSS selector or XPath
            by: Locator strategy
            
        Returns:
            str: Element text or None if not found
        """
        try:
            element = self.wait.until(EC.presence_of_element_located((by, selector)))
            return element.text
        except (TimeoutException, NoSuchElementException):
            return None
    
    def scroll_to_element(self, selector: str, by: By = By.CSS_SELECTOR) -> bool:
        """
        Scroll to element.
        
        Args:
            selector: CSS selector or XPath
            by: Locator strategy
            
        Returns:
            bool: True if scroll successful, False otherwise
        """
        try:
            element = self.wait.until(EC.presence_of_element_located((by, selector)))
            self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
            time.sleep(1)  # Allow scroll to complete
            return True
        except (TimeoutException, NoSuchElementException) as e:
            self.logger.error(f"Failed to scroll to element {selector}: {e}")
            return False


class ActionSequenceGenerator:
    """Generate action sequences for n8n workflow replication."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_trigger_node(self, node_name: str, node_type: str = "Webhook", config: Dict = None) -> Dict:
        """
        Create trigger node action.
        
        Args:
            node_name: Name of the trigger node
            node_type: Type of trigger (Webhook, Manual, etc.)
            config: Node configuration
            
        Returns:
            Dict: Action sequence entry
        """
        return {
            "type": "trigger",
            "name": node_name,
            "node_type": node_type,
            "config": config or {},
            "position": {"x": 100, "y": 100}
        }
    
    def create_action_node(self, node_name: str, node_type: str, config: Dict = None) -> Dict:
        """
        Create action node.
        
        Args:
            node_name: Name of the action node
            node_type: Type of action (OpenAI, HTTP Request, etc.)
            config: Node configuration
            
        Returns:
            Dict: Action sequence entry
        """
        return {
            "type": "action",
            "name": node_name,
            "node_type": node_type,
            "config": config or {},
            "position": {"x": 300, "y": 100}
        }
    
    def create_connection(self, from_node: int, to_node: int) -> Dict:
        """
        Create connection between nodes.
        
        Args:
            from_node: Source node index
            to_node: Target node index
            
        Returns:
            Dict: Connection action
        """
        return {
            "type": "connect",
            "from": from_node,
            "to": to_node,
            "connection_type": "data"
        }
    
    def generate_workflow_sequence(self, nodes: List[Dict], connections: List[Dict]) -> List[Dict]:
        """
        Generate complete workflow action sequence.
        
        Args:
            nodes: List of node definitions
            connections: List of connection definitions
            
        Returns:
            List[Dict]: Complete action sequence
        """
        sequence = []
        
        # Add node creation actions
        for i, node in enumerate(nodes):
            sequence.append({
                "step": i + 1,
                "action": "create_node",
                "node": node
            })
        
        # Add connection actions
        for i, connection in enumerate(connections):
            sequence.append({
                "step": len(nodes) + i + 1,
                "action": "connect_nodes",
                "connection": connection
            })
        
        self.logger.info(f"Generated workflow sequence with {len(sequence)} steps")
        return sequence


def setup_logging(log_level: str = "INFO", log_file: str = "automation.log") -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Log file path
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def save_action_sequence(sequence: List[Dict], output_path: str) -> bool:
    """
    Save action sequence to JSON file.
    
    Args:
        sequence: Action sequence list
        output_path: Output file path
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sequence, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Failed to save action sequence: {e}")
        return False


def load_action_sequence(input_path: str) -> Optional[List[Dict]]:
    """
    Load action sequence from JSON file.
    
    Args:
        input_path: Input file path
        
    Returns:
        List[Dict]: Action sequence or None if failed
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load action sequence: {e}")
        return None

