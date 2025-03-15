import socket
import nmap

# Port Scanning
def scan_network(target):
    """Scan the network for open ports"""
    nm = nmap.PortScanner()
    nm.scan(target, '1-1024')  # Scanning ports 1 to 1024
    results = []
    print(nm.all_hosts())
    for proto in nm[target].all_protocols():
        lport = nm[target][proto].keys()
        for port in lport:
            state = nm[target][proto][port]['state']
            results.append({"port": port, "state": state})
            print(f"Port: {port}\tState: {state}")
            
    return results

# Information Gathering
def get_info(domain):
    """Get the IP address of a given domain"""
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
    'scan_network': scan_network
}