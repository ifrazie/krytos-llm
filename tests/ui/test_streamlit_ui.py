import pytest
from unittest.mock import patch
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os
import time
import subprocess
import signal

from app import main

class TestStreamlitUI:
    """Test the Streamlit UI using Selenium (headless browser testing)"""
    
    @pytest.fixture(scope="class")
    def setup_streamlit(self):
        """Setup Streamlit server for testing"""
        # Start Streamlit server in background
        cmd = ['streamlit', 'run', 'app.py', '--server.port=8501', '--server.headless=true']
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Allow time for server to start
        time.sleep(5)
        
        # Setup webdriver
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
        # Yield for tests to use
        yield driver
        
        # Teardown
        driver.quit()
        os.kill(process.pid, signal.SIGTERM)
    
    @pytest.mark.ui
    @patch('app.get_ollama_models')
    def test_app_title_and_models(self, mock_get_models, setup_streamlit):
        """Test that the app title and model selection appear correctly"""
        # Skip this test if environment not set for UI testing
        if os.environ.get('SKIP_UI_TESTS') == 'true':
            pytest.skip("Skipping UI tests based on environment configuration")
            
        # Configure mock
        mock_get_models.return_value = ["llama3.1:latest", "mistral:latest"]
        
        # Get the driver from fixture
        driver = setup_streamlit
        
        # Navigate to Streamlit app
        driver.get("http://localhost:8501")
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "h1"))
        )
        
        # Check title
        title = driver.find_element(By.TAG_NAME, "h1").text
        assert "Krytos AI Security Analyst" in title
        
        # Check if sidebar exists
        sidebar = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'stSidebar')]"))
        )
        assert sidebar is not None
    
    @pytest.mark.ui
    @patch('app.load_tool_configs')
    @patch('app.get_ollama_models')
    def test_function_calling_toggle(self, mock_get_models, mock_load_tools, setup_streamlit):
        """Test that the function calling toggle works"""
        # Skip this test if environment not set for UI testing
        if os.environ.get('SKIP_UI_TESTS') == 'true':
            pytest.skip("Skipping UI tests based on environment configuration")
            
        # Configure mocks
        mock_get_models.return_value = ["llama3.1:latest"]
        mock_load_tools.return_value = [{"type": "function", "function": {"name": "get_info"}}]
        
        # Get the driver from fixture
        driver = setup_streamlit
        
        # Navigate to Streamlit app
        driver.get("http://localhost:8501")
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "h1"))
        )
        
        # Find and click the toggle for function calling
        toggle = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//div[contains(@data-testid, 'stToggleButton')]"))
        )
        toggle.click()
        
        # Check for success message after toggle
        success_msg = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'success')]"))
        )
        assert success_msg is not None