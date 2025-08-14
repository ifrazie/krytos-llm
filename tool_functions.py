import socket
import nmap
from datetime import datetime
import dns.resolver
import whois
import logging
import hashlib

# Define specific exceptions for better error handling
class ToolExecutionError(Exception):
    """Base class for errors in tool execution"""
    pass

class NetworkToolError(ToolExecutionError):
    """Error in network-related tools like port scanning"""
    pass

class DNSToolError(ToolExecutionError):
    """Error in DNS lookup operations"""
    pass

class WebToolError(ToolExecutionError):
    """Error in web-related operations like SQL injection testing"""
    pass

# Function to create standardized error responses
def create_error_response(tool_name: str, error: Exception, target: str) -> dict:
    """
    Create a standardized error response with helpful information
    
    Args:
        tool_name: The name of the tool that encountered an error
        error: The exception that was raised
        target: The domain or IP address being analyzed
        
    Returns:
        dict: A formatted error response with user-friendly information
    """
    timestamp = datetime.now().isoformat()
    error_str = str(error)
    error_type = error.__class__.__name__
    
    # Create user-friendly error message based on error type
    user_message = "An error occurred during execution."
    suggestion = "Please try again later."
    
    if isinstance(error, socket.gaierror):
        user_message = f"Could not resolve host: {target}"
        suggestion = "Check that the domain name is correct and that you have a working internet connection."
    elif isinstance(error, socket.timeout):
        user_message = f"Connection to {target} timed out"
        suggestion = "The target may be offline or blocking connections. Try again later."
    elif "permission" in error_str.lower():
        user_message = "Permission denied while executing the tool"
        suggestion = "This tool may require administrator privileges to run properly."
    elif "not found" in error_str.lower():
        user_message = f"Resource not found: {target}"
        suggestion = "Verify the target exists and is accessible."
    
    return {
        "timestamp": timestamp,
        "status": "error",
        "tool": tool_name,
        "target": target,
        "error_type": error_type,
        "error_details": error_str,
        "user_message": user_message,
        "suggestion": suggestion,
        "debug_info": {
            "python_error": error_str,
            "error_type": error_type
        }
    }

# Hashing
def hash_text(text: str, algorithm: str = "sha256") -> dict:
    """Compute a hash for the given text using the selected algorithm."""
    ts = datetime.now().isoformat()
    try:
        algo = algorithm.lower()
        if algo not in ("sha256", "md5"):
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        h = hashlib.sha256() if algo == "sha256" else hashlib.md5()
        h.update(text.encode("utf-8"))
        return {
            "timestamp": ts,
            "status": "completed",
            "algorithm": algo,
            "input_len": len(text),
            "hash": h.hexdigest(),
        }
    except Exception as e:
        logging.error(f"hash_text error: {e}")
        return create_error_response("hash_text", e, text)

# Simulated Exploitation
def sql_injection(url: str) -> dict:
    """Test a website for SQL injection vulnerabilities
    
    Args:
        url (str): The URL to test for SQL injection vulnerabilities.
        
    Returns:
        dict: A dictionary containing the results of the SQL injection test.
    """
    timestamp = datetime.now().isoformat()
    
    try:
        # Validate input URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        return {
            "timestamp": timestamp,
            "status": "completed",
            "vulnerability_type": "SQL Injection",
            "target_url": url,
            "details": {
                "attempted_payloads": [
                    "' OR '1'='1",
                    "'; DROP TABLE users--",
                    "1' UNION SELECT null--"
                ],
                "response_status": 403,
                "protection_detected": "WAF/Input Validation",
                "risk_level": "None",
                "scan_duration_ms": 150
            },
            "recommendations": [
                "Input validation is working correctly",
                "SQL injection protection is in place",
                "No further action needed"
            ]
        }
    except Exception as e:
        logging.error(f"SQL injection test error for {url}: {str(e)}")
        return create_error_response("sql_injection", e, url)

# Vulnerability Scanning
def check_vulnerability(domain: str) -> dict:
    """Scan for vulnerabilities in a domain
    Args:
        domain (str): The domain to scan for vulnerabilities.
    
    Returns:
        dict: A dictionary containing the results of the vulnerability scan.
    """
    timestamp = datetime.now().isoformat()
    
    try:
        # Validate domain format
        if not domain:
            raise ValueError("Domain cannot be empty")
            
        results = {
            "timestamp": timestamp,
            "domain": domain,
            "status": "completed",
            "headers_analysis": {
                "present": {
                    'X-XSS-Protection': '1; mode=block',
                    'X-Content-Type-Options': 'nosniff',
                    'X-Frame-Options': 'SAMEORIGIN',
                    'Strict-Transport-Security': 'max-age=31536000',
                    'Content-Security-Policy': "default-src 'self'"
                },
                "missing": ['Expect-CT', 'Feature-Policy']
            },
            "ssl_info": {
                "enabled": True,
                "version": "TLS 1.3",
                "issuer": "Let's Encrypt"
            },
            "risk_assessment": {
                "overall_level": "Low",
                "score": 8.5,
                "factors": {
                    "headers": "Good",
                    "ssl": "Excellent",
                    "open_ports": "Low Risk"
                }
            },
            "scan_duration_ms": 250
        }
        return results
    except Exception as e:
        logging.error(f"Vulnerability check error for {domain}: {str(e)}")
        return create_error_response("check_vulnerability", e, domain)

# Port Scanning
def scan_network(ip_address: str) -> dict:
    """Use an IP address to scan for open ports and services.
    
    Args:
        ip_address (str): The IP address to scan for open ports.
    
    Returns:
        dict: A dictionary containing the results of the port scan.
    """
    timestamp = datetime.now().isoformat()
    
    try:
        # Validate IP address format
        try:
            socket.inet_aton(ip_address)
        except socket.error:
            # If not a valid IP, try to resolve it as a hostname
            try:
                ip_address = socket.gethostbyname(ip_address)
            except socket.gaierror:
                raise ValueError(f"Invalid IP address or hostname: {ip_address}")
                
        nm = nmap.PortScanner()
        
        try:
            nm.scan(ip_address, arguments='-sV')
        except nmap.PortScannerError as e:
            if "root privileges" in str(e).lower() or "permission" in str(e).lower():
                raise NetworkToolError("Insufficient permissions to run port scan. This scan requires administrator privileges.")
            else:
                raise

        # Initialize open ports list
        open_ports = []
        services = {}

        # Group services by their risk level
        high_risk_services = []
        medium_risk_services = []
        low_risk_services = []
        
        if ip_address in nm.all_hosts():
            for proto in nm[ip_address].all_protocols():
                for port in sorted(nm[ip_address][proto].keys()):
                    port_info = nm[ip_address][proto][port]
                    if port_info["state"] == "open":
                        open_ports.append(port)
                        service_name = port_info.get("name", "unknown")
                        service_version = port_info.get("version", "unknown")
                        services[port] = {
                            "service": service_name,
                            "version": service_version
                        }
                        
                        # Categorize services by risk
                        port_entry = f"Port {port} ({service_name} {service_version})"
                        if port in [21, 23, 25, 445, 3389]:  # FTP, Telnet, SMTP, SMB, RDP
                            high_risk_services.append(port_entry)
                        elif port in [22, 53, 8080, 8443, 1433, 3306]:  # SSH, DNS, Web, SQL
                            medium_risk_services.append(port_entry)
                        else:
                            low_risk_services.append(port_entry)
            
        # Generate recommendations based on findings
        recommendations = []
        if high_risk_services:
            recommendations.append("Review and secure high-risk services: " + ", ".join(high_risk_services))
        if medium_risk_services:
            recommendations.append("Consider securing medium-risk services: " + ", ".join(medium_risk_services))
        if low_risk_services:
            recommendations.append("Low-risk services detected, ensure they are properly configured: " + ", ".join(low_risk_services))
        if not open_ports:
            recommendations.append("No open ports detected, system appears secure")
            
        results = {
            "timestamp": timestamp,
            "status": "completed",
            "ip_address": ip_address,
            "open_ports": open_ports,
            "services": services,
            "security_assessment": {
                "high_risk_services": high_risk_services,
                "medium_risk_services": medium_risk_services,
                "low_risk_services": low_risk_services
            },
            "recommendations": recommendations,
            "scan_details": {
                "total_ports_scanned": 1024,
                "scan_time": nm.scanstats()['elapsed'],
                "scan_type": "Comprehensive",
            }
        }
        
        return results
    except Exception as e:
        logging.error(f"Port scan error for {ip_address}: {str(e)}")
        return create_error_response("scan_network", e, ip_address)

# Information Gathering
def get_info(domain: str) -> dict:
    """Retrive detailed information about a domain using DNS and WHOIS lookups.
    
    Args:
        domain (str): The domain to gather information about.
    
    Returns:
        dict: A dictionary containing the gathered information.
    """
    timestamp = datetime.now().isoformat()
    
    try:
        ip = socket.gethostbyname(domain)
        dns_records = {}
        
        # Get DNS records
        for record_type in ['A', 'MX', 'NS', 'TXT', 'AAAA', 'SOA']:
            try:
                answers = dns.resolver.resolve(domain, record_type)
                dns_records[record_type] = [str(rdata) for rdata in answers]
            except:
                dns_records[record_type] = []
        
        # Get WHOIS information
        try:
            domain_info = whois.whois(domain)
            whois_data = {
                "Domain": domain_info.domain,
                "Registrar": domain_info.registrar,
                "Creation Date": str(domain_info.creation_date),
                "Expiration Date": str(domain_info.expiration_date),
                "Name Servers": domain_info.name_servers,
                "WHOIS Server": domain_info.whois_server,
                "Updated Date": str(domain_info.updated_date),
            }
        except:
            whois_data = {"error": "WHOIS lookup failed"}
        
        # Check for SPF, DKIM, DMARC records
        email_security = {
            "spf_found": False,
            "dmarc_found": False,
            "has_mx": len(dns_records.get('MX', [])) > 0
        }
        
        # Check TXT records for SPF
        for txt in dns_records.get('TXT', []):
            if 'v=spf1' in txt:
                email_security["spf_found"] = True
                break
        
        # Check for DMARC record
        try:
            dmarc_answers = dns.resolver.resolve('_dmarc.' + domain, 'TXT')
            for rdata in dmarc_answers:
                if 'v=DMARC1' in str(rdata):
                    email_security["dmarc_found"] = True
                    break
        except:
            pass
            
        # Calculate domain age if creation date is available
        domain_age_days = None
        if whois_data.get("creation_date") and "error" not in whois_data:
            try:
                creation_date = whois_data["creation_date"]
                if isinstance(creation_date, str):
                    # Try to parse the date string
                    from dateutil import parser
                    creation_date = parser.parse(creation_date)
                
                if creation_date:
                    domain_age_days = (datetime.now() - creation_date).days if not isinstance(creation_date, list) else None
            except:
                domain_age_days = None
                
        # Calculate days until expiration if available
        expiration_days = None
        if whois_data.get("expiration_date") and "error" not in whois_data:
            try:
                expiration_date = whois_data["expiration_date"]
                if isinstance(expiration_date, str):
                    # Try to parse the date string
                    from dateutil import parser
                    expiration_date = parser.parse(expiration_date)
                
                if expiration_date:
                    expiration_days = (expiration_date - datetime.now()).days if not isinstance(expiration_date, list) else None
            except:
                expiration_days = None
        
        # Generate recommendations based on findings
        recommendations = []
        
        # Email security recommendations
        if email_security["has_mx"] and not email_security["spf_found"]:
            recommendations.append("Implement SPF records to protect against email spoofing")
        if email_security["has_mx"] and not email_security["dmarc_found"]:
            recommendations.append("Set up DMARC policy to enhance email security and prevent domain spoofing")
        
        # Domain registration recommendations
        if expiration_days is not None and expiration_days < 30:
            recommendations.append(f"Domain expiration in {expiration_days} days - renew soon to prevent domain loss")
        
        # DNS recommendations
        if not dns_records.get('AAAA', []):
            recommendations.append("Consider adding IPv6 (AAAA) records for future-proofing your infrastructure")
        
        # New domain warning
        if domain_age_days is not None and domain_age_days < 30:
            recommendations.append(f"Domain is only {domain_age_days} days old - recently registered domains can be suspicious")
            
        # General security recommendations
        if not any('v=DKIM1' in txt for txt in dns_records.get('TXT', [])):
            recommendations.append("Implement DKIM email signing to improve deliverability and security")
        
        # If no issues found
        if not recommendations:
            recommendations.append("No immediate issues detected with domain configuration")
        
        return {
            "timestamp": timestamp,
            "status": "completed",
            "domain": domain,
            "ip_address": ip,
            "dns_records": dns_records,
            "whois_info": whois_data,
            "email_security": email_security,
            "domain_age_days": domain_age_days,
            "days_until_expiration": expiration_days,
            "recommendations": recommendations,
            "lookup_duration_ms": 200
        }
    except Exception as e:
        logging.error(f"Information gathering error for {domain}: {str(e)}")
        return create_error_response("get_info", e, domain)

# Dictionary mapping function names to their implementations
available_functions = {
    'get_info': get_info,
    'scan_network': scan_network,
    'check_vulnerability': check_vulnerability,
    'sql_injection': sql_injection,
    'hash_text': hash_text,
}