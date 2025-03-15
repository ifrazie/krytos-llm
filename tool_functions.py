import socket

def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

def subtract_two_numbers(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b

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
    'add_two_numbers': add_two_numbers,
    'subtract_two_numbers': subtract_two_numbers,
    'get_info': get_info
}