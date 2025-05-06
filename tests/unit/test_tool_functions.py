import pytest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import the functions to be tested
from tool_functions import get_info, scan_network, check_vulnerability, sql_injection

class TestToolFunctions:
    """Test the tool functions implemented in tool_functions.py"""
    
    @patch('socket.gethostbyname')
    @patch('dns.resolver.resolve')
    @patch('whois.whois')
    def test_get_info_success(self, mock_whois, mock_dns_resolve, mock_gethostbyname):
        """Test the get_info function with successful responses"""
        # Setup mocks
        mock_gethostbyname.return_value = "192.168.1.1"
        
        # Mock DNS resolver
        mock_dns_answer = MagicMock()
        mock_dns_answer.__str__.return_value = "test.example.com"
        mock_dns_resolve.return_value = [mock_dns_answer]
        
        # Mock WHOIS data
        mock_whois_data = MagicMock()
        mock_whois_data.registrar = "Test Registrar"
        mock_whois_data.creation_date = datetime(2020, 1, 1)
        mock_whois_data.expiration_date = datetime(2026, 1, 1)
        mock_whois.return_value = mock_whois_data
        
        # Call the function
        result = get_info("example.com")
        
        # Assertions
        assert result["status"] == "completed"
        assert result["domain"] == "example.com"
        assert result["ip_address"] == "192.168.1.1"
        assert "dns_records" in result
        assert "whois_info" in result
        assert result["whois_info"]["registrar"] == "Test Registrar"
        
    @patch('socket.gethostbyname')
    def test_get_info_error(self, mock_gethostbyname):
        """Test the get_info function with an error response"""
        # Setup mocks to raise an exception
        mock_gethostbyname.side_effect = Exception("Test exception")
        
        # Call the function
        result = get_info("invalid-domain.com")
        
        # Assertions
        assert result["status"] == "error"
        assert "error" in result
        assert result["domain"] == "invalid-domain.com"
    
    @patch('nmap.PortScanner')
    def test_scan_network(self, mock_port_scanner):
        """Test the scan_network function"""
        # Setup mock scanner
        scanner_instance = MagicMock()
        mock_port_scanner.return_value = scanner_instance
        
        # Configure scan method
        scanner_instance.scan = MagicMock()
        
        # Configure all_hosts method
        scanner_instance.all_hosts.return_value = ["192.168.1.1"]
        
        # Configure protocols
        scanner_instance.__getitem__.return_value.all_protocols.return_value = ["tcp"]
        
        # Configure port keys
        port_mock = MagicMock()
        port_mock.keys.return_value = [22, 80, 443]
        scanner_instance.__getitem__.return_value.__getitem__.return_value = port_mock
        
        # Configure port info for specific ports
        port_info_22 = {"state": "open", "name": "ssh", "version": "OpenSSH 8.9"}
        port_info_80 = {"state": "open", "name": "http", "version": "nginx 1.18.0"}
        port_info_443 = {"state": "open", "name": "https", "version": "nginx 1.18.0"}
        
        port_mock.__getitem__.return_value = port_info_22  # Default response
        
        # Define a side effect to return different values based on port number
        def get_port_info(port):
            if port == 22:
                return port_info_22
            elif port == 80:
                return port_info_80
            elif port == 443:
                return port_info_443
            return {}
        
        port_mock.__getitem__.side_effect = get_port_info
        
        # Configure scanstats
        scanner_instance.scanstats.return_value = {"elapsed": "10.5"}
        
        # Call the function
        result = scan_network("192.168.1.1")
        
        # Assertions
        assert result["status"] == "completed"
        assert result["ip_address"] == "192.168.1.1"
        assert 22 in result["open_ports"]
        assert 80 in result["open_ports"]
        assert 443 in result["open_ports"]
        assert "ssh" in result["services"][22]["service"]
    
    def test_check_vulnerability(self):
        """Test the check_vulnerability function"""
        result = check_vulnerability("example.com")
        
        # Assertions for simulated vulnerability check
        assert result["status"] == "completed"
        assert result["domain"] == "example.com"
        assert "headers_analysis" in result
        assert "ssl_info" in result
        assert "risk_assessment" in result
    
    def test_sql_injection(self):
        """Test the sql_injection function"""
        result = sql_injection("https://example.com/login")
        
        # Assertions for simulated SQL injection test
        assert result["status"] == "completed"
        assert result["vulnerability_type"] == "SQL Injection"
        assert result["target_url"] == "https://example.com/login"
        assert "details" in result
        assert "recommendations" in result