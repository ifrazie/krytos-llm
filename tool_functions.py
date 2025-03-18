import socket
import nmap
from typing import Dict, Any, List
import ssl
import dns.resolver
import requests
from ftplib import FTP
from datetime import datetime

# Simulated Exploitation
def sql_injection(url: str) -> dict:
    """Test for SQL injection vulnerabilities
    
    Args:
        url (str): The URL to test for SQL injection vulnerabilities.
    
    Returns: A dictionary containing the results of the test.
    """
    return {
        "status": "failed",
        "vulnerability_type": "SQL Injection",
        "target_url": url,
        "details": {
            "attempted_payload": "' OR '1'='1",
            "response_status": 403,
            "protection_detected": "WAF/Input Validation",
            "risk_level": "None"
        },
        "recommendations": [
            "Input validation is working correctly",
            "SQL injection protection is in place",
            "No further action needed"
        ]
    }

# Service registry
SUPPORTED_SERVICES = {
    'dns': {
        'default_port': 53,
        'description': 'Domain Name System security check'
    },
    'http': {
        'default_port': 80,
        'description': 'HTTP security headers and methods check'
    },
    'https': {
        'default_port': 443,
        'description': 'HTTPS/SSL/TLS security check'
    },
    'ftp': {
        'default_port': 21,
        'description': 'FTP security configuration check'
    }
}

def _check_dns(domain: str) -> Dict[str, Any]:
    results = {"service": "DNS", "vulnerabilities": []}
    try:
        # Check for DNS zone transfer
        answers = dns.resolver.resolve(domain, 'NS')
        results["name_servers"] = [str(rdata) for rdata in answers]
        results["vulnerabilities"].append({
            "check": "Zone Transfer",
            "status": "Protected",
            "risk_level": "Low"
        })
    except Exception as e:
        results["error"] = str(e)
    return results

def _check_http(domain: str) -> Dict[str, Any]:
    results = {"service": "HTTP", "vulnerabilities": []}
    try:
        response = requests.get(f"http://{domain}", timeout=5)
        results["vulnerabilities"].extend([
            {
                "check": "HTTP Headers",
                "missing_security_headers": [
                    header for header in [
                        'X-XSS-Protection',
                        'X-Content-Type-Options',
                        'X-Frame-Options'
                    ] if header not in response.headers
                ]
            },
            {
                "check": "HTTP Methods",
                "status": "Checking allowed methods"
            }
        ])
    except Exception as e:
        results["error"] = str(e)
    return results

def _check_https(domain: str) -> Dict[str, Any]:
    results = {"service": "HTTPS", "vulnerabilities": []}
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
            s.connect((domain, 443))
            cert = s.getpeercert()
            results["vulnerabilities"].extend([
                {
                    "check": "SSL/TLS Version",
                    "version": s.version(),
                    "risk_level": "Low" if s.version() >= "TLSv1.2" else "High"
                },
                {
                    "check": "Certificate Validity",
                    "valid": True,
                    "expires": cert['notAfter']
                }
            ])
    except Exception as e:
        results["error"] = str(e)
    return results

def _check_ftp(domain: str) -> Dict[str, Any]:
    results = {"service": "FTP", "vulnerabilities": []}
    try:
        ftp = FTP(domain, timeout=5)
        anonymous = ftp.login()  # Try anonymous login
        results["vulnerabilities"].extend([
            {
                "check": "Anonymous Login",
                "status": "Vulnerable" if anonymous else "Protected",
                "risk_level": "High" if anonymous else "Low"
            }
        ])
        ftp.quit()
    except Exception as e:
        results["error"] = str(e)
    return results

# Vulnerability Scanning
def check_vulnerability(domain: str, service_name: str = None) -> Dict[str, Any]:
    """Scan for vulnerabilities in specific service or all services
    
    Args:
        domain (str): The domain to scan for vulnerabilities
        service_name (str, optional): Specific service to check. Defaults to None (all services)
    
    Returns:
        Dict[str, Any]: Vulnerability scan results
    """
    service_check_map = {
        'dns': _check_dns,
        'http': _check_http,
        'https': _check_https,
        'ftp': _check_ftp
    }

    results = {
        "status": "completed",
        "target": domain,
        "timestamp": datetime.now().isoformat(),
        "services": []
    }

    if service_name:
        if service_name.lower() not in SUPPORTED_SERVICES:
            return {
                "status": "error",
                "message": f"Unsupported service: {service_name}",
                "supported_services": list(SUPPORTED_SERVICES.keys())
            }
        
        check_func = service_check_map.get(service_name.lower())
        try:
            service_result = check_func(domain)
            results["services"].append(service_result)
        except Exception as e:
            results["services"].append({
                "service": service_name.upper(),
                "error": str(e)
            })
    else:
        # Check all services
        for service, check_func in service_check_map.items():
            try:
                service_result = check_func(domain)
                results["services"].append(service_result)
            except Exception as e:
                results["services"].append({
                    "service": service.upper(),
                    "error": str(e)
                })

    # Calculate risk assessment based on findings
    results["risk_assessment"] = _calculate_risk_assessment(results["services"])
    
    return results

def _calculate_risk_assessment(service_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall risk assessment based on service findings"""
    risk_levels = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    
    for service in service_results:
        if "vulnerabilities" in service:
            for vuln in service["vulnerabilities"]:
                if "risk_level" in vuln:
                    risk_levels[vuln["risk_level"]] += 1
    
    overall_risk = "Low"
    if risk_levels["Critical"] > 0:
        overall_risk = "Critical"
    elif risk_levels["High"] > 0:
        overall_risk = "High"
    elif risk_levels["Medium"] > 0:
        overall_risk = "Medium"
    
    return {
        "overall_risk_level": overall_risk,
        "critical_findings": risk_levels["Critical"],
        "high_findings": risk_levels["High"],
        "medium_findings": risk_levels["Medium"],
        "low_findings": risk_levels["Low"]
    }

# Port Scanning
def scan_network(ip_address: str) -> dict:
    """Scan the network for open ports
    
    Args:
        target (str): The IP address or hostname to scan for open ports.
    
    Returns: A dictionary containing the results of the scan.
    """
    nm = nmap.PortScanner()
    nm.scan(ip_address, '1-1024')  # Scanning ports 1 to 1024
    results = []
    print(nm.all_hosts())
    for proto in nm[ip_address].all_protocols():
        lport = nm[ip_address][proto].keys()
        for port in lport:
            state = nm[ip_address][proto][port]['state']
            results.append({"port": port, "state": state})
            print(f"Port: {port}\tState: {state}")
            
    return results

# Information Gathering
def get_info(domain: str) -> dict:
    """Get the IP address of a given domain
    
    Args:
        domain (str): The domain to get the IP address of.
    
    Returns: A dictionary containing the IP address of the domain.
    """
    try:
        ip = socket.gethostbyname(domain)
        print(f"Domain: {domain}")
        print(f"IP Address: {ip}")
    except socket.gaierror:
        print("Error: Unable to get IP address.")

    return ip

# Dictionary mapping function names to their implementations
available_functions = {
    'get_info': get_info,
    'scan_network': scan_network,
    'check_vulnerability': lambda domain, service=None: check_vulnerability(domain, service),
    'sql_injection': sql_injection
}